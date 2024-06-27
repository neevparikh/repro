from utils import parser

from inference import inference
from train import train

if __name__ == "__main__":
    args = parser()
    if args.subparser_name == "inference":
        inference(args)
    elif args.subparser_name == "train":
        train(args)
    else:
        raise ValueError(args.subparser_name)
