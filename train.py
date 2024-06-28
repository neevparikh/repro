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
            max_length=Config.context_size
            + 1,  # we add one since we shift the training and target by one
        ),
        batched=True,
    )
    dataset = dataset.with_format("torch")
    training_data = dataset["train"]

    dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)  # type: ignore

    model = Transformer()
    model.to(get_device(args))

    for batch in dataloader:
        tokens, padding_mask = batch["input_ids"], batch["attention_mask"]
        input_tokens = tokens[:

