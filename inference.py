from argparse import Namespace

import torch
import tqdm
from pprint import pprint

from config import Config
from model import Transformer
from utils import get_device, setup_tokenizer


def inference(args: Namespace) -> None:
    model = Transformer()
    model.eval()
    model.to(get_device(args))

    tokenizer = setup_tokenizer(args)

    tokens: torch.Tensor = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding=Config.token_padding,
        truncation=Config.token_truncation,
        max_length=Config.context_size,
    )["input_ids"]

    with torch.no_grad():
        output = tokens.repeat(args.num_completions, 1)  # B x T

        for _ in tqdm.trange(
            0, min(args.max_tokens, Config.context_size - len(args.prompt))
        ):
            logits = model(output)  # B x T x V
            probs = logits[:, -1, :].softmax(-1)  # B x 1 x V
            probs, idxs = torch.topk(probs, args.topk, dim=-1)  # (B x K, B x K)
            sampled_indices = torch.multinomial(probs, 1)  # B x 1

            # we do this since the samples are indicies into the top-k, not into the vocab V
            # if K == V then this is not needed
            next_token = torch.gather(
                idxs, -1, sampled_indices
            )  # indexing into (B x K) on K dim by B x 1

            output = torch.cat((output, next_token), dim=-1)  # B x (T + 1)

    decoded = tokenizer.batch_decode(output)

    pprint(decoded, width=120)
