from dataclasses import dataclass


@dataclass
class Config:
    # model configs
    context_size: int = 1094
    vocab_size: int = 50257
    dim_model: int = 256
    dim_mlp: int = 512
    heads: int = 4
    layers: int = 3
    mlp_dropout: float = 0.1
    attention_dropout: float = 0.1
    # token configs
    token_padding: str = "longest"
    token_truncation: str = "longest_first"
