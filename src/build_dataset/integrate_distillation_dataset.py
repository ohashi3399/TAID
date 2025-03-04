import os
from datasets import load_dataset, Dataset, DatasetDict
from collections import defaultdict


def create_and_push_dataset(data, dataset_name):
    """
    dictのリストからHugging Face Datasetを作成し、push_to_hubする。

    Args:
        data (list[dict]): "messages"と"source"カラムを含むdictのリスト。
        dataset_name (str): push_to_hubする際のデータセット名。
        hub_token (str): Hugging Face Hubのトークン。
    """

    # sourceごとにデータをグループ化
    grouped_data = defaultdict(list)
    for item in data:
        grouped_data[item["source"]].append(item)

    train_data = []
    test_data = []

    # sourceごとにtrainとtestに分割
    for source, items in grouped_data.items():
        if len(items) <= 2000:
            test_data.extend(items)
        else:
            test_data.extend(items[:2000])
            train_data.extend(items[2000:])

    # DatasetDictを作成
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data)
    })

    # push_to_hub
    dataset_dict.push_to_hub(dataset_name, token=os.environ.get("HUGGINGFACE_API_TOKEN"))


ds_name = "ryota39/wildchat-en-ja-540k"
ds = load_dataset(ds_name, split="train")

corpus = list()
for record in ds:
    data_point = dict()
    if len(corpus) > 22000 - 1:
        break
    data_point["messages"] = record["messages"]
    data_point["source"] = ds_name
    corpus.append(data_point)


ds_name = "ryota39/open-math-instruct2-cot-ja-portion"
ds = load_dataset(ds_name, split="train")
for record in ds:
    data_point = dict()
    if len(corpus) > 44000 - 1:
        break
    messages = [
        {"role": "user", "content": record["problem_ja"]},
        {"role": "user", "content": record["generated_solution_cot"]},
    ]
    data_point["messages"] = messages
    data_point["source"] = ds_name
    corpus.append(data_point)


ds_name = "ryota39/waka-500m-dataset"
ds = load_dataset(ds_name, split="train")
for record in ds:
    data_point = dict()
    if len(corpus) > 66000 - 1:
        break
    data_point["messages"] = record["chosen"]
    data_point["source"] = ds_name
    corpus.append(data_point)


# データセット名とHugging Face Hubのトークンを設定
dataset_name = "ryota39/taid-dataset"

# データセットを作成してpush_to_hub
create_and_push_dataset(corpus, dataset_name)
