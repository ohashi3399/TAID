import os
from typing import List
from inspect import isfunction
import itertools
import glob
from natsort import natsorted

import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file as load_safetensors


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def flatten_list(x):
    return list(itertools.chain.from_iterable(x))


def get_generated_ids(generated_ids: torch.Tensor, input_ids: torch.Tensor):
    input_len = input_ids.shape[-1]
    return generated_ids[:, input_len:]


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_decay_parameter_names(model) -> List[str]:
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    no_decay = ["bias"]
    decay_parameters = [
        name for name in decay_parameters if not any(nd in name for nd in no_decay)
    ]
    return decay_parameters


def get_optimizer_params(model: nn.Module, loss_fn: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    decay_parameters = get_decay_parameter_names(model)

    if loss_fn is not None:
        param_optimizer += list(loss_fn.named_parameters())
        decay_parameters += get_decay_parameter_names(loss_fn)

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_grouped_parameters


def load_tokenizer(tokenizer_path: str, **tokenizer_kwargs):
    if "phi-3" in tokenizer_path.lower():
        tokenizer_kwargs["pad_token"] = "<unk>"
        tokenizer_kwargs["padding_side"] = "right"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_best_checkpoint_name(logdir):
    ckpt = os.path.join(logdir, "last**.ckpt")
    ckpt = natsorted(glob.glob(ckpt))
    if len(ckpt) == 0:
        ckpt = os.path.join(logdir, "epoch**.ckpt")
        ckpt = natsorted(glob.glob(ckpt))
    ckpt = ckpt[-1]
    return ckpt


def load_state_dict(ckpt):
    def get_state_dict_from_lightning(path):
        pl_sd = torch.load(path, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        return sd

    print(f"Loading model from {ckpt}")
    if ckpt.endswith("ckpt"):
        if os.path.isdir(ckpt) and os.path.exists(
            os.path.join(ckpt, "pytorch_model.bin")
        ):
            sd = torch.load(os.path.join(ckpt, "pytorch_model.bin"), map_location="cpu")
        elif os.path.isdir(ckpt):
            # convert deepspeed checkpoint to fp32 state dict
            import tempfile
            from lightning.pytorch.utilities.deepspeed import (
                convert_zero_checkpoint_to_fp32_state_dict,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                fp32_ckpt = os.path.join(tmpdir, "pytorch_model.bin")
                convert_zero_checkpoint_to_fp32_state_dict(ckpt, fp32_ckpt)
                sd = get_state_dict_from_lightning(fp32_ckpt)
        else:
            sd = get_state_dict_from_lightning(ckpt)
    elif ckpt.endswith("safetensors"):
        sd = load_safetensors(ckpt)
    else:
        raise NotImplementedError
    return sd


def load_hf_model_from_config(
    model_path,
    ckpt,
    model_name="student_model",
    vocab_size=None,
):
    if vocab_size is not None:
        # load config
        model_config = AutoConfig.from_pretrained(model_path)
        model_config.vocab_size = vocab_size
    sd = load_state_dict(ckpt)
    sd = {k.replace(f"{model_name}.", ""): v for k, v in sd.items()}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        state_dict=sd,
    )
    return model
