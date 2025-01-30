from typing import Union, Dict, Optional, List
import random
from collections import namedtuple, deque

import numpy as np
import torch
from torch import nn
from transformers import GenerationConfig
from lightning import LightningModule
from src.utils import default


class ReplayBuffer:
    def __init__(self, capacity: int = 1000, field_names: Optional[List] = None):
        self.field_names = default(
            field_names, ["input_ids", "attention_mask", "labels"]
        )
        self.replay_memory = deque(maxlen=capacity)
        self.data = namedtuple(
            "Generation",
            field_names=self.field_names,
        )

    def __len__(self):
        return len(self.replay_memory)

    def sample(self, batch_size: int):
        data = random.sample(self.replay_memory, k=batch_size)
        res = {}
        for k in self.field_names:
            res[k] = torch.stack([getattr(d, k) for d in data], dim=0)
        return res

    def move_to_device(self, model_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)
        return model_data

    def move_to_memory(self, model_data):
        device = torch.device("cpu")
        model_data_cpu = {}
        for k in model_data:
            model_data_cpu[k] = model_data[k].to(device)

        for idx in range(model_data_cpu["input_ids"].size(0)):
            e = self.data(*[model_data_cpu[k][idx] for k in self.field_names])
            self.replay_memory.append(e)


def run_sample(model, gen_data, pad_token_id, generation_config, return_ids=False):
    bs = gen_data["input_ids"].size(0)
    max_length = gen_data["input_ids"].size(1) + generation_config.max_new_tokens
    results = {
        "input_ids": torch.ones(
            bs, max_length, dtype=torch.long, device=gen_data["input_ids"].device
        )
        * pad_token_id,
        "attention_mask": torch.zeros(
            bs, max_length, dtype=torch.float, device=gen_data["input_ids"].device
        ),
        "labels": torch.ones(
            bs, max_length, dtype=torch.long, device=gen_data["input_ids"].device
        )
        * -100,
    }

    full_ids = model.generate(
        **gen_data,
        generation_config=generation_config,
    )
    input_ids = full_ids[:, : gen_data["input_ids"].size(1)]
    response_ids = full_ids[:, gen_data["input_ids"].size(1) :]

    for i in range(len(input_ids)):
        result_id = torch.cat(
            (
                input_ids[i][input_ids[i] != pad_token_id],
                response_ids[i][response_ids[i] != pad_token_id],
            ),
        )
        input_id = input_ids[i][input_ids[i] != pad_token_id]
        response_id = response_ids[i][response_ids[i] != pad_token_id]

        results["input_ids"][i, : len(result_id)] = result_id
        results["labels"][i, len(input_id) : len(result_id)] = response_id
    results["attention_mask"] = torch.where(results["input_ids"] != pad_token_id, 1, 0)
    results["attention_mask"] = results["attention_mask"].long()
    results["labels"] = results["labels"].long()
    if return_ids:
        return results, {"full_ids": full_ids, "response_ids": response_ids}
    return results


class SampleGenerator(nn.Module):
    def __init__(
        self,
        sampling_type: str = "adaptive",
        replay_ratio: str = "decreasing",
        mixed_alpha: float = 0.5,
        adaptive_threshold: float = 0.0,
        loss_eps: float = 0.0,
        capacity: int = 1000,
        model_ratio: Optional[float] = None,
    ):
        super().__init__()
        assert sampling_type in ["mixed", "adaptive"]
        self.sampling_type = sampling_type
        assert replay_ratio in ["constant", "increasing", "decreasing"]
        self.replay_ratio = replay_ratio
        self.mixed_alpha = mixed_alpha
        self.loss_eps = loss_eps
        self.capacity = capacity
        self.replay_buffer = ReplayBuffer(capacity)
        self.prev_loss = None
        self.model_ratio = model_ratio
        self.register_buffer("adaptive_threshold", torch.tensor(adaptive_threshold))

    def update(self, loss: Union[float, torch.Tensor] = None):
        if self.sampling_type == "adaptive":
            assert loss is not None
            if self.prev_loss is None:
                self.prev_loss = loss
            elif loss >= self.prev_loss + self.loss_eps:
                self.adaptive_threshold += 0.1
                self.adaptive_threshold = min(self.adaptive_threshold, 1.0)
                self.prev_loss = loss

    def get_model_inputs_gen(self, batch):
        gen_data = batch.get("model_inputs_gen")
        gen_data = {
            k: v for k, v in gen_data.items() if k in ["input_ids", "attention_mask"]
        }
        return gen_data

    @torch.no_grad()
    def forward(
        self,
        lightning_module: LightningModule,
        batch: Dict[str, Dict[str, torch.Tensor]],
        generation_config: GenerationConfig,
        global_step: int,
        total_iters: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        r = np.random.uniform(0, 1)
        if self.replay_ratio == "constant":
            samp_threshold = self.adaptive_threshold * 0.5
        elif self.replay_ratio == "increasing":
            samp_threshold = self.adaptive_threshold * global_step / total_iters
        else:
            samp_threshold = self.adaptive_threshold * (1 - global_step / total_iters)

        model_batch = None
        # data generation from student models
        if not lightning_module.training:
            # no sampling during eval
            model_batch = None
        elif self.sampling_type == "mixed" and r < self.mixed_alpha:
            gen_data = self.get_model_inputs_gen(batch)
            bsz = gen_data["input_ids"].size(0)

            model_batch = run_sample(
                lightning_module.student_model,
                gen_data,
                pad_token_id=lightning_module.tokenizer.pad_token_id,
                generation_config=generation_config,
            )
            self.replay_buffer.move_to_memory(model_batch)
            model_batch = self.replay_buffer.sample(bsz)
            model_batch = self.replay_buffer.move_to_device(
                model_batch, lightning_module.student_model.device
            )

        elif self.sampling_type == "adaptive" and (
            r < samp_threshold
            or (r < self.adaptive_threshold and len(self.replay_buffer) < self.capacity)
        ):
            gen_data = self.get_model_inputs_gen(batch)
            bsz = gen_data["input_ids"].size(0)

            model_batch = run_sample(
                lightning_module.student_model,
                gen_data,
                pad_token_id=lightning_module.tokenizer.pad_token_id,
                generation_config=generation_config,
            )
            self.replay_buffer.move_to_memory(model_batch)

        elif self.sampling_type == "adaptive" and r < self.adaptive_threshold:
            gen_data = self.get_model_inputs_gen(batch)
            bsz = gen_data["input_ids"].size(0)
            model_batch = self.replay_buffer.sample(bsz)
            model_batch = self.replay_buffer.move_to_device(
                model_batch, lightning_module.student_model.device
            )
        elif self.model_ratio is not None:
            # replace data samples with teacher samples
            assert "model_inputs_from_model" in batch
            inputs = batch.pop("model_inputs_from_model")
            if self.model_ratio > np.random.uniform(0, 1):
                model_batch = inputs
            else:
                del inputs

        # update batch
        if model_batch is not None:
            batch["model_inputs"] = model_batch
        return batch


def get_sampler(args):
    if args.sampling_type is None:
        return None
    else:
        return SampleGenerator(
            sampling_type=args.sampling_type,
        )
