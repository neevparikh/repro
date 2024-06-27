from numerize.numerize import numerize
import torch
from transformers import AutoTokenizer

from argparse import ArgumentParser, Namespace

from model.modules import Transformer, Config


def setup_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.vocab_size == Config.vocab_size
    return tokenizer


def get_device(args: Namespace) -> str:
    return "cuda" if torch.cuda.is_available() and args.gpu else "cpu"


def run(args: Namespace) -> None:
    model = Transformer()
    model.eval()
    model.to(get_device(args))
    print(f"Model parameters: {numerize(sum([p.numel() for p in model.parameters()]))}")

    tokenizer = setup_tokenizer()

    assert args.chat is None, "Cannot do --chat yet"

    tokens = tokenizer(
        args.prompt,
        return_tensors="np",
        padding=Config.token_padding,
        truncation=Config.token_truncation,
        max_length=Config.context_size,
    )

    __import__("pdb").set_trace()


if __name__ == "__main__":
    parser = ArgumentParser(prog="GPT2", description="Reproduction of GPT2")
    parser.add_argument("--checkpoint-path", help="Pass in trained model")
    parser.add_argument("--gpu", help="Run on inference on GPU", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt", help="Run inference on this prompt")
    group.add_argument(
        "--chat", help="Run inference on this chat, use jsonl messages format"
    )
    args = parser.parse_args()
    run(args)
