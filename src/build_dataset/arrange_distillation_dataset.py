import os
from datasets import load_dataset, Dataset, DatasetDict


ds = load_dataset("ryota39/wildchat-en-ja-540k", split="train")

corpus = list()
for data_point in ds:
    if len(corpus) > 20000 - 1:
        break
    corpus.append(data_point)

train = corpus[:10000]
test = corpus[10000:]

new_ds = DatasetDict(
    {
        "train": Dataset.from_list(train),
        "test": Dataset.from_list(test),
    }
)

new_ds.push_to_hub(
    "ryota39/wildchat-10k",
    private=True,
    token=os.environ.get("HUGGINGFACE_API_KEY"),
)
