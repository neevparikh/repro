from argparse import Namespace

from torch.utils.data import DataLoader

from config import Config
from model import Transformer
from utils import get_dataset, get_device, setup_tokenizer


def train(args: Namespace) -> None:
    dataset = get_dataset(args)
    tokenizer = setup_tokenizer(args)

    dataset = dataset.map(
        lambda data: tokenizer(
            data["text"],
            return_tensors="pt",
            padding=Config.token_padding,
            truncation=Config.token_truncation,
            max_length=Config.context_size,
        ),
        batched=True,
    )
    dataset = dataset.with_format("torch")
    training_data = dataset["train"]

    dataloader = DataLoader(training_data, batch_size=args.batch_size)  # type: ignore

    for batch in dataloader:
        __import__("pdb").set_trace()

    model = Transformer()
    model.eval()
    model.to(get_device(args))
