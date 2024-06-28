from argparse import Namespace

import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from config import Config


def setup_tokenizer(args: Namespace):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    assert tokenizer.vocab_size == Config.vocab_size
    if args.subparser_name == "inference":
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
    return tokenizer


def get_device(args: Namespace) -> str:
    return "cuda" if torch.cuda.is_available() and args.gpu else "cpu"


def get_dataset(args: Namespace) -> DatasetDict:
    assert args.small
    dataset = load_dataset("stas/openwebtext-10k")
    assert isinstance(dataset, DatasetDict)
    return dataset
