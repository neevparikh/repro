from math import sqrt

from numerize.numerize import numerize
import torch
import torch.nn as nn

from config import Config


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(dim))
        self.b_2 = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super(MultiHeadAttention, self).__init__()
        assert Config.dim_model % Config.heads == 0
        self.dim_model = Config.dim_model
        self.heads = Config.heads
        self.dim_per_head = Config.dim_model // Config.heads
        self.dim_per_head_sqrt = sqrt(self.dim_per_head)
        self.context_size = Config.context_size

        self.output_linear = nn.Linear(self.dim_model, self.dim_model)
        self.combined_linear = nn.Linear(self.dim_model, self.dim_model * 3)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.context_size, self.context_size)).view(
                1, 1, self.context_size, self.context_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_sequence_length, embedding_dim = x.size()
        assert embedding_dim == self.dim_model

        (query, key, value) = self.combined_linear(x).split(
            self.dim_model, dim=2
        )  # All are (B x T x M)

        queries = query.view(
            batch_size, token_sequence_length, self.heads, self.dim_per_head
        ).transpose(
            1, 2
        )  # (B x H x T x M/H)
        keys = key.view(
            batch_size, token_sequence_length, self.heads, self.dim_per_head
        ).transpose(
            1, 2
        )  # (B x H x T x M/H)
        values = value.view(
            batch_size, token_sequence_length, self.heads, self.dim_per_head
        ).transpose(
            1, 2
        )  # (B x H x T x M/H)

        compatibility_scores = (
            torch.matmul(queries, keys.transpose(-2, -1)) / self.dim_per_head_sqrt
        )  # (B x H x T x M/H) @ (B x H x M/H x T) -> B x H x T x T

        compatibility_scores = compatibility_scores.masked_fill(
            self.mask[:, :, :token_sequence_length, :token_sequence_length] == 0,
            float("-inf"),
        )

        p_attn = compatibility_scores.softmax(dim=-1)

        attended_output = torch.matmul(
            p_attn, values
        )  # (B x H x T x T) @ (B x H x T x M/H) -> (B x H x T x M/H)

        attended_output = (
            attended_output.transpose(1, 2)  # (B x T x H x M/H)
            .contiguous()
            .view(batch_size, token_sequence_length, self.dim_model)
        )  # (B x T x M)

        return self.output_linear(attended_output)  # (B x T x M)


class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.w1 = nn.Linear(Config.dim_model, Config.dim_mlp)
        self.w2 = nn.Linear(Config.dim_mlp, Config.dim_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)  # (B x T x ML)
        x = nn.functional.gelu(x)
        return self.w2(x)  # (B x T x M)


class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention()
        self.attention_dropout = nn.Dropout(p=Config.attention_dropout)
        self.attention_layer_norm = LayerNorm(Config.dim_model)

        self.mlp = MLP()
        self.mlp_dropout = nn.Dropout(p=Config.mlp_dropout)
        self.mlp_layer_norm = LayerNorm(Config.dim_model)

    def forward(self, x) -> torch.Tensor:
        x = self.attention_layer_norm(x)
        z = self.attention(x)  # (B x T x M)
        z = x + self.attention_dropout(z)

        z = self.mlp_layer_norm(z)
        r = self.mlp(z)  # (B x T x M)
        r = z + self.mlp_dropout(r)

        return r  # (B x T x M)


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(Config.layers)])
        self.last_norm = LayerNorm(Config.dim_model)

    def forward(self, x) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.last_norm(x)  # (B x T x M)


class Embeddings(nn.Module):
    def __init__(self) -> None:
        super(Embeddings, self).__init__()
        self.token_embeddings = nn.Embedding(Config.vocab_size, Config.dim_model)
        self.positional_embeddings = nn.Embedding(Config.vocab_size, Config.dim_model)

    def forward(self, indices) -> torch.Tensor:
        _, token_sequence_length = indices.size()  # (B x T)
        tokens = self.token_embeddings(indices)  # (B x T x M)
        positions = self.positional_embeddings(
            torch.arange(
                0, token_sequence_length, dtype=torch.long, device=indices.device
            )
        )  # (T x M)
        return tokens + positions  # (B x T x M)


class Transformer(nn.Module):
    def __init__(self, Config=Config) -> None:
        super(Transformer, self).__init__()
        self.embeddings = Embeddings()
        self.decoder = Decoder()
        self.prediction_head = nn.Linear(
            Config.dim_model, Config.vocab_size, bias=False
        )
        print(
            f"Model parameters: {numerize(sum([p.numel() for p in self.parameters()]))}"
        )

    def forward(self, tokens) -> torch.Tensor:
        embeddings = self.embeddings(tokens)  # (B x T x M)
        attended_embeddings = self.decoder(embeddings)  # (B x T x M)
        logits = self.prediction_head(attended_embeddings)  # (B x T x V)
        return logits
