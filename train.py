"""
WikiLM Training
===============
Full training loop with cosine LR scheduling, mixed precision,
gradient clipping, checkpointing, and validation.
"""

import os
import sys
import math
import time
import argparse

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from config import get_model_config, TrainConfig, ModelConfig
from model import WikiLM
from tokenizer import BPETokenizer
from dataset import load_wikipedia_articles, tokenize_corpus, create_dataloaders


def get_device(preference: str = "auto") -> torch.device:
    """Determine the best available device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def get_lr(step: int, total_steps: int, config: TrainConfig) -> float:
    """Cosine annealing learning rate with linear warmup."""
    warmup_steps = int(total_steps * config.warmup_ratio)

    if step < warmup_steps:
        # linear warmup
        return config.learning_rate * (step + 1) / warmup_steps

    # cosine decay
    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def evaluate(model: WikiLM, val_loader, device: torch.device) -> float:
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(1, n_batches)


def save_checkpoint(
    model: WikiLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    config: ModelConfig,
    path: str,
):
    """Save a training checkpoint."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "config": config,
        },
        path,
    )


def train(model_config_name: str = "small", train_config: TrainConfig = None):
    """Main training function."""
    if train_config is None:
        train_config = TrainConfig()

    model_config = get_model_config(model_config_name)
    device = get_device(train_config.device)
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print(f"WikiLM Training — {model_config_name} config")
    print(f"Device: {device}")
    print("=" * 60)

    # ── Load or train tokenizer ──────────────────────────────────────────────
    if os.path.exists(train_config.tokenizer_path):
        print(f"\nLoading tokenizer from {train_config.tokenizer_path}")
        tokenizer = BPETokenizer.load(train_config.tokenizer_path)
    else:
        print("\nNo tokenizer found. Training a new one...")
        from tokenizer import fetch_wikipedia_texts
        texts = fetch_wikipedia_texts(train_config.num_articles)
        tokenizer = BPETokenizer()
        tokenizer.train(texts, vocab_size=model_config.vocab_size)
        tokenizer.save(train_config.tokenizer_path)

    # update vocab size to match tokenizer
    model_config.vocab_size = tokenizer.vocab_size

    # ── Load and tokenize data ───────────────────────────────────────────────
    articles = load_wikipedia_articles(train_config.num_articles)
    token_ids = tokenize_corpus(articles, tokenizer, tokenizer_path=train_config.tokenizer_path)

    train_loader, val_loader = create_dataloaders(
        token_ids,
        context_length=model_config.context_length,
        batch_size=train_config.batch_size,
        val_split=train_config.val_split,
        num_workers=train_config.num_workers,
    )

    # ── Initialize model ─────────────────────────────────────────────────────
    model = WikiLM(model_config).to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Config: {model_config}")

    # optionally compile for faster training (PyTorch 2.0+)
    if train_config.compile_model and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── Optimizer ────────────────────────────────────────────────────────────
    # separate weight decay: apply to weight matrices, skip biases and layernorms
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
    )

    use_amp = train_config.mixed_precision and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # ── Training loop ────────────────────────────────────────────────────────
    total_steps = train_config.epochs * len(train_loader)
    global_step = 0
    best_val_loss = float("inf")

    print(f"\nStarting training: {train_config.epochs} epochs, {total_steps} total steps")
    print("-" * 60)

    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # update learning rate
            lr = get_lr(global_step, total_steps, train_config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # forward pass with optional mixed precision
            with autocast(device.type, enabled=use_amp):
                _, loss = model(x, y)

            # backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            # logging
            if global_step % train_config.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                elapsed = time.time() - epoch_start
                tokens_per_sec = (batch_idx + 1) * train_config.batch_size * model_config.context_length / elapsed
                print(
                    f"  Step {global_step:6d} | "
                    f"Epoch {epoch + 1}/{train_config.epochs} | "
                    f"Loss {loss.item():.4f} (avg {avg_loss:.4f}) | "
                    f"LR {lr:.2e} | "
                    f"{tokens_per_sec:,.0f} tok/s"
                )

            # evaluation
            if global_step % train_config.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"  ── Val loss: {val_loss:.4f}", end="")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(
                        train_config.checkpoint_dir,
                        f"wikilm_{model_config_name}_best.pt",
                    )
                    save_checkpoint(model, optimizer, epoch, global_step, val_loss, model_config, best_path)
                    print(f" ★ New best! Saved to {best_path}")
                else:
                    print()

            # periodic save
            if global_step % train_config.save_interval == 0:
                ckpt_path = os.path.join(
                    train_config.checkpoint_dir,
                    f"wikilm_{model_config_name}_step{global_step}.pt",
                )
                save_checkpoint(model, optimizer, epoch, global_step, loss.item(), model_config, ckpt_path)
                print(f"  ── Checkpoint saved to {ckpt_path}")

        # end-of-epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1} complete — avg loss: {avg_epoch_loss:.4f}, time: {epoch_time:.1f}s")

        # save epoch checkpoint
        epoch_path = os.path.join(
            train_config.checkpoint_dir,
            f"wikilm_{model_config_name}_epoch{epoch + 1}.pt",
        )
        save_checkpoint(model, optimizer, epoch, global_step, avg_epoch_loss, model_config, epoch_path)

    print("\n" + "=" * 60)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WikiLM")
    parser.add_argument("--config", type=str, default="small", choices=["small", "medium", "large"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_articles", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    tc = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_articles=args.num_articles,
        learning_rate=args.lr,
        device=args.device,
    )

    train(model_config_name=args.config, train_config=tc)
