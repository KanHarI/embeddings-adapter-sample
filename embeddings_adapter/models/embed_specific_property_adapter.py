import dataclasses
from typing import Callable

import torch

from embeddings_adapter.models.mlp import MLPConfig, MLP
from embeddings_adapter.models.new_gelu import new_gelu


@dataclasses.dataclass
class EmbedSpecificCategoryAdapterConf:
    n_layers: int
    n_embed: int
    dropout: float
    inference_alpha: float
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    dtype: torch.dtype
    device: str
    init_std: float


class EmbedSpecificCategoryAdapter:
    def __init__(self, config: EmbedSpecificCategoryAdapterConf):
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
