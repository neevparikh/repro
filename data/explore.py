from argparse import Namespace
from src.utils import get_dataset, setup_tokenizer

dataset = get_dataset(Namespace(small=True))
train = dataset["train"]
tokenizer = setup_tokenizer(Namespace(subparser_name="explore"))
__import__("pdb").set_trace()
