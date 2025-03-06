import argparse
import os
from transformers import AutoTokenizer
from datasets import load_dataset
from litdata import optimize
from functools import partial

MODELS = {
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "stablelm": "stabilityai/stablelm-zephyr-3b",
    "calm3": "cyberagent/calm3-22b-chat",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
}
MAX_LENGTH = 2048
MAX_OUTPUT_LENGTH = 512


def tokenize(example, tokenizer):
    column = "messages" if "messages" in example else "chosen"
    text = tokenizer.apply_chat_template(
        example[column], tokenize=False, add_generation_prompt=False
    )
    messages = text.split(generation_prompt)
    input_text = generation_prompt.join(messages[:-1]) + generation_prompt
    output_text = messages[-1]
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    res = {"model_inputs": {"input_ids": input_ids, "labels": input_ids.clone()}}

    gen_input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    res["model_inputs_gen"] = {"input_ids": gen_input_ids}
    res["response"] = output_text
    return res


def filter_length(example, max_input_len, max_output_len):
    max_length = max_input_len + max_output_len
    if example["model_inputs"]["input_ids"].size(1) > max_length:
        return False
    if example["model_inputs_gen"]["input_ids"].size(1) > max_input_len:
        return False
    output_tokens = tokenizer(example["response"], return_tensors="pt").input_ids
    if output_tokens.size(1) > max_output_len:
        return False
    return True


def fn(index, data):
    yield data[index]


def prepare_train(args, tokenizer):
    dataset = load_dataset(
        # "ryota39/wildchat-10k",
        "ryota39/taid-dataset",
        split="train",
        token=os.environ.get("HUGGINGFACE_API_KEY"),
    )
    column_names = list(dataset.features)
    dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=args.num_proc,
        desc="Applying chat template",
        remove_columns=column_names,
    )
    dataset = dataset.with_format("torch")
    dataset = dataset.filter(
        filter_length,
        fn_kwargs={
            "max_input_len": MAX_LENGTH - MAX_OUTPUT_LENGTH,
            "max_output_len": MAX_OUTPUT_LENGTH,
        },
        num_proc=args.num_proc,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    optimize(
        fn=partial(fn, data=dataset),
        inputs=list(range(len(dataset))),
        output_dir=os.path.join(args.output_dir, args.model_type, "train"),
        num_workers=16,
        chunk_bytes="500MB",
    )


def prepare_test(args, tokenizer):
    dataset = load_dataset(
        # "ryota39/wildchat-10k",
        "ryota39/taid-dataset",
        split="test",
        token=os.environ.get("HUGGINGFACE_API_KEY"),
    )
    column_names = list(dataset.features)
    dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=args.num_proc,
        desc="Applying chat template",
        remove_columns=column_names,
    )
    dataset = dataset.with_format("torch")
    dataset = dataset.filter(
        filter_length,
        fn_kwargs={
            "max_input_len": MAX_LENGTH - MAX_OUTPUT_LENGTH,
            "max_output_len": MAX_OUTPUT_LENGTH,
        },
        num_proc=args.num_proc,
    )
    ds = dataset.train_test_split(test_size=2000, seed=42, shuffle=True)
    dataset = ds["test"]

    os.makedirs(args.output_dir, exist_ok=True)

    optimize(
        fn=partial(fn, data=dataset),
        inputs=list(range(len(dataset))),
        output_dir=os.path.join(args.output_dir, args.model_type, "test"),
        num_workers=2,
        chunk_bytes="500MB",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        choices=list(MODELS.keys()),
        default="phi-3",
        help="Teacher type",
    )
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--num_proc", type=int, default=64, help="number of workers for processing"
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model_type])
    if args.model_type == "phi-3":
        # https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/sample_finetune.py#L141
        tokenizer.pad_token = (
            tokenizer.unk_token
        )  # use unk rather than eos token to prevent endless generation
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = "right"

    if args.model_type in ["phi-3", "stablelm"]:
        generation_prompt = "<|assistant|>\n"
    elif args.model_type in ["llama-2"]:
        generation_prompt = " [/INST] "
    elif args.model_type == "calm3":
        generation_prompt = "<|im_start|>assistant\n"
    elif args.model_type == "qwen2.5":
        generation_prompt = "<|im_start|>assistant\n"
    else:
        raise NotImplementedError(args.model_type)
    prepare_train(args, tokenizer)
    prepare_test(args, tokenizer)
