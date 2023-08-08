import dataclasses
from typing import Callable

import torch

from embeddings_adapter.models.new_gelu import new_gelu


@dataclasses.dataclass
class EmbedColorExperimentConfig:
    _dtype: str
    wandb_log: bool
    run_name: str
    batch_size: int
    log_interval: int
    eval_interval: int
    eval_iters: int
    max_steps: int
    lr: float
    momentum: float
    n_layers: int
    n_embed: int
    dropout: float
    inference_alpha: float
    linear_size_multiplier: int
    _activation: str
    device: str
    init_std: float
    ln_eps: float
    triplet_loss_bias: float
    norm_loss_cost: float
    warmup_iters: int

    @property
    def dtype(self) -> torch.dtype:
        match self._dtype:
            case "float32":
                return torch.float32
            case "float64":
                return torch.float64
            case "float16":
                return torch.float16
            case "bfloat16":
                return torch.bfloat16
            case _:
                raise ValueError(f"Unknown dtype: {self._dtype}")

    @property
    def activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        match self._activation:
            case "relu":
                return torch.nn.functional.relu
            case "new_gelu":
                return new_gelu
            case _:
                raise ValueError(f"Unknown activation: {self._activation}")
