from typing import Optional
import torch
from math import sqrt
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, dim_model) -> None:
        super()
        assert dim_model % heads == 0
        self.dim_model = dim_model
        self.heads = heads
        self.dim_per_head = dim_model // heads
        self.dim_per_head_sqrt = sqrt(self.dim_per_head)

        self.output_linear = nn.Linear(self.dim_model, self.dim_model)
        self.query_linear = nn.Linear(self.dim_model, self.dim_model)
        self.key_linear = nn.Linear(self.dim_model, self.dim_model)
        self.value_linear = nn.Linear(self.dim_model, self.dim_model)

        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        query = (
            self.query_linear(query)
            .view(batch_size, -1, self.heads, self.dim_per_head)
            .transpose(1, 2)
        )
        values = (
            self.values_linear(values)
            .view(batch_size, -1, self.heads, self.dim_per_head)
            .transpose(1, 2)
        )
        keys = (
            self.keys_linear(keys)
            .view(batch_size, -1, self.heads, self.dim_per_head)
            .transpose(1, 2)
        )

        compatibility_scores = (
            torch.matmul(query, keys.transpose(-2, -1)) / self.dim_per_head_sqrt
        )

        if mask:
            mask = mask.unsqueeze(1)
            compatibility_scores = compatibility_scores.masked_fill(mask == 0, -1e9)

        p_attn = self.dropout(compatibility_scores.softmax(dim=-1))

        attended_output = torch.matmul(p_attn, values)
        attended_output = (
            attended_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.dim_model)
        )

        return self.output_linear(attended_output)


class PointwiseFeedForward(nn.Module):
    def __init__(self, dim_model, dim_feedforward) -> None:
        super()
        self.w1 = nn.Linear(dim_model, dim_feedforward)
        self.w2 = nn.Linear(dim_feedforward, dim_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.w1(x).relu()))


class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super()
        dim_model = 128
        dim_mlp = 512

        self.attention = MultiHeadAttention(heads=2, dim_model=dim_model)
        self.attention_dropout = nn.Dropout(p=0.1)
        self.attention_layer_norm = LayerNorm(dim_model)

        self.mlp = nn.Sequential(
            nn.Linear(dim_model, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim_model),
        )
        self.mlp_dropout = nn.Dropout(p=0.1)
        self.mlp_layer_norm = LayerNorm(dim_model)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(dim=-1)

        x = self.attention_layer_norm(x)
        z, _ = self.attention(x, x, x, mask=mask)
        z = x + self.attention_dropout(z)

        z = self.mlp_layer_norm(z)
        r = self.mlp(z)
        r = z + self.mlp_dropout(r)

        return r


class Encoder(nn.Module):
    def __init__(self) -> None:
        super()
