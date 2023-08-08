import torch


def norm_loss(original_batch: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    return torch.abs((original_batch**2).mean() - (output**2).mean())
