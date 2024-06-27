from argparse import Namespace

from config import Config
from model import Transformer
from utils import get_device, setup_tokenizer


def inference(args: Namespace) -> None:
    model = Transformer()
    model.eval()
    model.to(get_device(args))

    tokenizer = setup_tokenizer(args)

    tokens = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding=Config.token_padding,
        truncation=Config.token_truncation,
        max_length=Config.context_size,
    )
