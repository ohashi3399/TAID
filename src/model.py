from typing import Dict, Any
import torch
from torch import Tensor
import lightning as L
from transformers import get_scheduler, GenerationConfig, AutoModelForCausalLM
from src.metrics import compute_metrics
from src.loss import get_loss_fn, LossOutput
from src.sampler import get_sampler
from src.utils import default, flatten_list, get_generated_ids, get_optimizer_params


def initialize_generation_config(tokenizer, generation_config):
    assert tokenizer is not None, "You must define tokenizer before generation!"
    generation_config = GenerationConfig(**generation_config)
    generation_config.return_dict_in_generate = False
    if not generation_config.eos_token_id:
        generation_config.eos_token_id = tokenizer.eos_token_id
    if not generation_config.pad_token_id:
        generation_config.pad_token_id = default(
            tokenizer.pad_token_id, tokenizer.eos_token_id
        )
    return generation_config


class KDForLM(L.LightningModule):
    def __init__(self, args, tokenizer, generation_config=None):
        super().__init__()
        self.student_model = None
        self.teacher_model = None
        self.loss_fn = get_loss_fn(args)
        self.sampler = get_sampler(args)
        self.generation_config = default(
            generation_config,
            {"do_sample": False, "num_beams": 1, "max_new_tokens": 512},
        )
        self.tokenizer = tokenizer
        self.args = args
        self.validation_step_outputs = {}

    def configure_model(self):
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.args.student_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.student_model.resize_token_embeddings(len(self.tokenizer))
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.args.teacher_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.teacher_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, batch: Dict[str, Tensor], **kwargs) -> LossOutput:
        outputs: LossOutput = self.loss_fn(lightning_module=self, batch=batch, **kwargs)
        return outputs

    def shard_step(self, batch, step="train", prefix="", **kwargs):
        outputs = self(batch, **kwargs)
        loss_dict = outputs.loss_dict
        loss_dict = {
            f"{step}/{prefix}{k}": v for k, v in loss_dict.items() if v is not None
        }
        return outputs.loss, loss_dict

    def get_batch_size(self, batch) -> int:
        return batch["model_inputs"]["input_ids"].size(0)

    def get_num_tokens(self, batch) -> int:
        return torch.sum(
            batch["model_inputs"]["input_ids"] != self.tokenizer.pad_token_id
        )

    def sampling(self, batch):
        if self.sampler is None:
            return batch
        assert "model_inputs_gen" in batch
        # data generation from student model
        generation_config = initialize_generation_config(
            self.tokenizer, self.generation_config
        )
        batch = self.sampler(
            self,
            batch,
            generation_config,
            self.global_step,
            self.trainer.estimated_stepping_batches,
        )
        return batch

    def training_step(self, batch, batch_idx) -> Tensor:
        batch_size = self.get_batch_size(batch)
        # sampling step
        batch = self.sampling(batch)
        # compute loss
        loss, loss_dict = self.shard_step(batch=batch)
        self.log_dict(loss_dict, batch_size=batch_size, prog_bar=True)
        return loss

    def generate(self, model_inputs_gen):
        """
        Generate from student
        """
        assert "response" in model_inputs_gen and isinstance(
            model_inputs_gen["response"][0], str
        )
        response = model_inputs_gen.pop("response")
        generation_config = initialize_generation_config(
            self.tokenizer, self.generation_config
        )
        # generate
        generated_ids = self.student_model.generate(
            **model_inputs_gen,
            generation_config=generation_config,
        )
        # extract generated ids
        generated_ids = get_generated_ids(generated_ids, model_inputs_gen["input_ids"])
        generated_answers = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return generated_answers, response

    def _compute_metric(self, outputs, step="val"):
        preds = flatten_list([item["preds"] for item in outputs])
        target = flatten_list([item["target"] for item in outputs])
        metric_res = compute_metrics(preds, target)
        metric_res = {f"{step}/{k}": v for k, v in metric_res.items()}
        return metric_res

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        res = {}
        batch_size = self.get_batch_size(batch)
        # sampling step
        batch = self.sampling(batch)

        loss, loss_dict = self.shard_step(batch=batch, step=f"val_{dataloader_idx}")

        self.log_dict(
            loss_dict,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        res["loss"] = loss
        if "model_inputs_gen" in batch:
            model_inputs_gen = batch.pop("model_inputs_gen")
            preds, target = self.generate(model_inputs_gen)
            res["preds"] = preds
            res["target"] = target
        if dataloader_idx not in self.validation_step_outputs:
            self.validation_step_outputs[dataloader_idx] = []
        self.validation_step_outputs[dataloader_idx].append(res)
        return loss

    def on_validation_epoch_end(self):
        res = {}
        for (
            idx,
            step_outputs,
        ) in self.validation_step_outputs.items():
            losses = torch.stack([item["loss"] for item in step_outputs])
            eval_loss = losses.mean()
            if "preds" in step_outputs[0]:
                metric_res = self._compute_metric(step_outputs, step=f"val_{idx}")
                res.update(metric_res)
            if self.sampler is not None and idx == 0:
                self.sampler.update(loss=eval_loss)
                if self.sampler.sampling_type == "adaptive":
                    res["adaptive_threshold"] = self.sampler.adaptive_threshold
        self.log_dict(res, logger=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def on_save_checkpoint(self, checkpoint):
        sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        for key in list(sd.keys()):
            if "teacher_model." in key:
                del sd[key]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        sd = self.state_dict()
        for key in list(sd.keys()):
            if "teacher_model." in key:
                if "state_dict" in checkpoint:
                    checkpoint["state_dict"][key] = sd[key]
                else:
                    checkpoint[key] = sd[key]

    def configure_optimizers(self):
        params = get_optimizer_params(self.student_model, self.loss_fn)
        optimizer = torch.optim.AdamW(params, lr=self.args.lr)
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_warmup_steps=0,
        )
        self.print(
            f"Setting up scheduler (estimated_stepping_batches: {self.trainer.estimated_stepping_batches})..."
        )
        scheduler = [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        ]
        return [optimizer], scheduler
