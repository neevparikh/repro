from argparse import Namespace, ArgumentParser

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


def parser() -> Namespace:
    parser = ArgumentParser(prog="GPT2", description="Reproduction of GPT2")

    # common between train and inference
    parser.add_argument("--batch-size", help="Pass in trained model")
    parser.add_argument("--gpu", help="Run on inference on GPU", action="store_true")
    parser.add_argument("--checkpoint-path", help="Pass in trained model")

    subparsers = parser.add_subparsers(
        help="Training or inference", dest="subparser_name"
    )
    train = subparsers.add_parser("train", help="Training")
    train.add_argument("--small", help="Pass in trained model")

    inference = subparsers.add_parser("inference", help="Inference")
    inference.add_argument("--prompt", help="Run inference on this prompt")
    inference.add_argument(
        "--top-k", help="Top k highest probability tokens to sample from", default=40
    )

    return parser.parse_args()
