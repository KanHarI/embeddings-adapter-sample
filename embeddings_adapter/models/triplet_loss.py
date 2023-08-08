import torch


def triplet_loss(batch: torch.Tensor, triplet_loss_bias: float) -> torch.Tensor:
    B, N, E = batch.shape  # (batch_size, 3, embedding_size)
    assert N == 3
    anchor = batch[:, 0, :]  # (batch_size, embedding_size)
    positive = batch[:, 1, :]  # (batch_size, embedding_size)
    negative = batch[:, 2, :]  # (batch_size, embedding_size)
    positive_distance = torch.norm(anchor - positive, dim=1)  # (batch_size,)
    negative_distance = torch.norm(anchor - negative, dim=1)  # (batch_size,)
    return torch.nn.functional.relu(
        positive_distance - negative_distance + triplet_loss_bias
    ).mean()  # (batch_size,)
