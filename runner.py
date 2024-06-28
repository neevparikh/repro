from argparse import Namespace, ArgumentParser

from inference import inference
from train import train


def parser() -> Namespace:
    parser = ArgumentParser(prog="GPT2", description="Reproduction of GPT2")

    # common between train and inference
    parser.add_argument("--gpu", help="Run on inference on GPU", action="store_true")
    parser.add_argument(
        "--checkpoint-path",
        help="Pass in trained model for inference or save models here",
    )

    subparsers = parser.add_subparsers(
        help="Training or inference", dest="subparser_name"
    )

    # training
    train = subparsers.add_parser("train", help="Training")
    train.add_argument(
        "--small", help="use small 10k dataset for development", action="store_true"
    )
    train.add_argument(
        "--batch-size", help="Batch size for training", default=32, type=int
    )

    # inference
    inference = subparsers.add_parser("inference", help="Inference")
    inference.add_argument("--prompt", help="Run inference on this prompt")
    inference.add_argument(
        "--topk",
        help="Top k highest probability tokens to sample from",
        default=40,
        type=int,
    )
    inference.add_argument(
        "--max-tokens",
        help="Maximum tokens to generate to, will stop if context window is exceeded with prompt",
        default=1094,
        type=int,
    )
    inference.add_argument(
        "--num-completions",
        help="Number of completions to generate. Note, max tokens is per completion.",
        default=1,
        type=int,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    if args.subparser_name == "inference":
        inference(args)
    elif args.subparser_name == "train":
        train(args)
    else:
        raise ValueError(args.subparser_name)
