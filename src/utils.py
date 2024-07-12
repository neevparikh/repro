from argparse import Namespace
import itertools

import torch
import wandb
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from src.config import Config


def setup_tokenizer(args: Namespace):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    assert len(tokenizer) == Config.vocab_size

    if args.subparser_name == "inference":
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

    return tokenizer


def get_device(args: Namespace) -> str:
    return "cuda" if torch.cuda.is_available() and args.gpu else "cpu"


def get_dataset(args: Namespace) -> DatasetDict:

    if args.small:
        dataset = load_dataset("stas/openwebtext-10k", split="train")
    else:
        raise NotImplemented()

    assert isinstance(dataset, Dataset)
    dataset = dataset.train_test_split(test_size=0.05)
    assert isinstance(dataset, DatasetDict)
    return dataset


def process_and_tokenize(data, tokenizer):
    return tokenizer(
        data["text"],
        truncation=Config.token_truncation,
        max_length=Config.context_size
        + 1,  # we add one since we shift the training and target by one
    )


def pack_sequences(data, tokenizer):
    # we add to account for tokens and targets are offset by 1
    context_size = Config.context_size + 1
    for l in data["input_ids"]:
        l.append(tokenizer.eos_token_id)
    tokens = list(itertools.chain.from_iterable(data["input_ids"]))
    total_tokens = len(tokens)
    padding_required = total_tokens % context_size
    tokens += [Config.padding_idx] * padding_required
    output = []
    for i in range(0, len(tokens) // context_size):
        output.append(tokens[i : i + context_size])
    data["input_ids"] = output
    return data


def log(metrics, step_key="iteration"):
    logstr = ""
    for key, value in metrics.items():
        match value:
            case float():
                logstr += f"{key}: {value:6f}"
            case _:
                logstr += f"{key}: {value}"

        match key:
            case "time" | "val_time":
                logstr += "ms, "
            case _:
                logstr += ", "

    print(logstr, flush=True)
    wandb.log(metrics, step=metrics[step_key])
