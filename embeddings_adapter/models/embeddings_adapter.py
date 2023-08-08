import dataclasses
from typing import Callable

import torch

from embeddings_adapter.models.mlp import MLP, MLPConfig


@dataclasses.dataclass
class EmbeddingAdapterConf:
    n_layers: int
    n_embed: int
    dropout: float
    inference_alpha: float
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    dtype: torch.dtype
    device: str
    init_std: float
    ln_eps: float


class EmbeddingAdapter(torch.nn.Module):
    def __init__(self, config: EmbeddingAdapterConf):
        super().__init__()
        self.config = config
        self.mlps_config = MLPConfig(
            n_in=config.n_embed,
            n_out=config.n_embed,
            linear_size_multiplier=config.linear_size_multiplier,
            activation=config.activation,
            dtype=config.dtype,
            device=config.device,
            init_std=config.init_std,
            dropout=config.dropout,
        )
        self.mlps = torch.nn.ModuleList(
            [MLP(self.mlps_config) for _ in range(config.n_layers)]
        )
        self.layer_norms = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(config.n_embed, eps=config.ln_eps)
                for _ in range(config.n_layers)
            ]
        )
        self.final_ln = torch.nn.LayerNorm(config.n_embed, eps=config.ln_eps)
        self.final_amp = torch.nn.Parameter(
            torch.zeros(1, dtype=config.dtype, device=config.device, requires_grad=True)
        )

    def init_weights(self) -> None:
        for mlp in self.mlps:
            mlp.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x
        for i in range(self.config.n_layers):
            x = x + self.mlps[i](self.layer_norms[i](x))
        x = self.final_ln(x)
        x = x * torch.exp(self.final_amp)
        if self.training:
            return x
        else:
            return (
                original_x * (1 - self.config.inference_alpha)
                + x * self.config.inference_alpha
            )
