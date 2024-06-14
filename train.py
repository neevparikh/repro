from datasets import DatasetDict, load_dataset
from numerize.numerize import numerize
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.modules import Transformer, Config

dataset = load_dataset("stas/openwebtext-10k")
assert isinstance(dataset, DatasetDict)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
assert tokenizer.vocab_size == Config.vocab_size

dataset = dataset.map(
    lambda data: tokenizer(
        data["text"],
        return_tensors="np",
        padding="max_length",
        truncation="longest_first",
        max_length=Config.context_size,
    ),
    batched=True,
)
dataset = dataset.with_format("torch")
training_data = dataset["train"]

dataloader = DataLoader(training_data, batch_size=64)  # type: ignore

for batch in dataloader:
    __import__("pdb").set_trace()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Transformer()
model.eval()
model.to(device)
print(f"Model parameters: {numerize(sum([p.numel() for p in model.parameters()]))}")
