import sys

import torch
import torch.utils.data
import wandb

from embeddings_adapter.conf.embed_color_experiment_conf import (
    EmbedColorExperimentConfig,
)
from embeddings_adapter.data.colors_dataset import ColorsDataloader
from embeddings_adapter.data.colors_raw_data import TEST_SET, TRAIN_SET
from embeddings_adapter.data.openai_cache import GLOBAL_EMBEDDINGS_CACHE
from embeddings_adapter.models.embeddings_adapter import (
    EmbeddingAdapter,
    EmbeddingAdapterConf,
)
from embeddings_adapter.models.norm_loss import norm_loss
from embeddings_adapter.models.triplet_loss import triplet_loss

EXPERIMENT_CONFIG = EmbedColorExperimentConfig(
    wandb_log=True,
    run_name="tiny-dataset-alpha-0.5",
    batch_size=12,
    log_interval=5,
    eval_interval=50,
    eval_iters=500,
    max_steps=10_000,
    lr=1e-4,
    momentum=0.9,
    _dtype="float32",
    n_layers=2,
    n_embed=1536,  # Same as OpenAI's ada-text-embeddings-002
    dropout=0.75,
    inference_alpha=0.5,
    linear_size_multiplier=4,
    _activation="new_gelu",
    device="cpu",
    init_std=1e-3,
    ln_eps=1e-4,
    triplet_loss_bias=0.1,
    norm_loss_cost=0.01,
    warmup_iters=1_000,
)


def main() -> int:
    config = EXPERIMENT_CONFIG
    if config.wandb_log:
        wandb.init(project="embeddings-highligh-color", name=config.run_name)
        wandb.config.update(config)
    print("Creating dataset...")
    train_dataset = ColorsDataloader(TRAIN_SET)
    test_dataset = ColorsDataloader(TEST_SET)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
    )
    print("Creating model...")
    embeddings_adapter_config = EmbeddingAdapterConf(
        n_layers=config.n_layers,
        n_embed=config.n_embed,
        dropout=config.dropout,
        inference_alpha=config.inference_alpha,
        linear_size_multiplier=config.linear_size_multiplier,
        activation=config.activation,
        dtype=config.dtype,
        device=config.device,
        init_std=config.init_std,
        ln_eps=config.ln_eps,
    )
    model = EmbeddingAdapter(embeddings_adapter_config)
    print("Initializing model...")
    model.init_weights()
    print("Training...")
    optimizer = torch.optim.SGD(
        lr=config.lr, momentum=config.momentum, params=model.parameters()
    )
    model.train()
    training_losses = torch.zeros(
        (config.eval_interval,), device="cpu", dtype=torch.float32
    )
    training_losses += float("inf")  # First iteration has no loss
    lr = 0.0
    best_eval_loss = float("inf")
    for step in range(config.max_steps):
        if step % config.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_losses = torch.zeros(
                    (config.eval_iters,), device="cpu", dtype=torch.float32
                )
                eval_triplet_losses = torch.zeros(
                    (config.eval_iters,), device="cpu", dtype=torch.float32
                )
                eval_norm_losses = torch.zeros(
                    (config.eval_iters,), device="cpu", dtype=torch.float32
                )
                for j in range(config.eval_iters):
                    batch = next(iter(test_loader))  # (B, 3, E)
                    batch = batch.to(embeddings_adapter_config.device)
                    result = model(batch)
                    eval_triplet_loss = triplet_loss(
                        result, config.triplet_loss_bias
                    )
                    eval_norm_loss = norm_loss(batch, result) * config.norm_loss_cost
                    loss = eval_triplet_loss + eval_norm_loss
                    eval_losses[j] = loss
                    eval_triplet_losses[j] = eval_triplet_loss
                    eval_norm_losses[j] = eval_norm_loss
                if config.wandb_log:
                    wandb.log(
                        {
                            "eval_loss": eval_losses.mean(),
                            "eval_triplet_loss": eval_triplet_losses.mean(),
                            "eval_norm_loss": eval_norm_losses.mean(),
                            "training_loss": training_losses.mean(),
                            "lr": lr,
                        },
                        step=step,
                    )
                print(
                    f"Step: {step}, eval loss: {eval_losses.mean()}, training loss: {training_losses.mean()}, lr: {lr}"
                )
                eval_loss = eval_losses.mean().item()
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    print("Saving model...")
                    torch.save(model.state_dict(), "embeddings_adapter.pt")
                    print("Saving embeddings cache...")
            # Save all embeddings to disk for cache
            GLOBAL_EMBEDDINGS_CACHE.save_to_json()
            model.train()
        batch = next(iter(train_loader))  # (B, 3, E)
        batch = batch.to(embeddings_adapter_config.device)
        result = model(batch)
        loss = (
            triplet_loss(result, config.triplet_loss_bias)
            + norm_loss(batch, result) * config.norm_loss_cost
        )
        if step < config.warmup_iters:
            lr = config.lr * min(1.0, step / config.warmup_iters)
        else:
            lr = config.lr * (1.0 - step / config.max_steps)
        for parameters_group in optimizer.param_groups:
            parameters_group["lr"] = lr
        optimizer.zero_grad()
        loss.backward()  # type: ignore
        optimizer.step()
        training_losses[step % config.eval_interval] = loss.item()
        if step % config.log_interval == 0:
            print(f"Step: {step}, loss: {loss.item()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
