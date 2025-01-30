import argparse
import os
import glob
from natsort import natsorted
from safetensors.torch import load_file as load_safetensors
import torch
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from src.utils import load_tokenizer


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


def main(
    tokenizer_path: str,
    model_path: str,
    ckpt: str,
):
    if ckpt.endswith("/"):
        ckpt = ckpt[:-1]
    if os.path.isdir(ckpt) and not ckpt.endswith(".ckpt"):
        ckpt = get_best_checkpoint_name(ckpt)
    save_dir = os.path.join(ckpt, "hf_model")
    if os.path.exists(save_dir):
        print("Already exists")
        return
    tokenizer = load_tokenizer(tokenizer_path)
    model: PreTrainedModel = load_hf_model_from_config(
        model_path=model_path,
        ckpt=ckpt,
        # to resize embeddings
        vocab_size=len(tokenizer) if tokenizer is not None else None,
    )
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="teacher's tokenizer path"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="student model path"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    args = parser.parse_args()
    main(tokenizer_path=args.tokenizer_path, model_path=args.model_path, ckpt=args.ckpt)
