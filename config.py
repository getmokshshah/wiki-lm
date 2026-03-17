"""
WikiLM Configuration
====================
Model architectures and training hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer model configuration."""
    vocab_size: int = 8192
    context_length: int = 256
    n_layers: int = 6
    n_heads: int = 6
    embed_dim: int = 384
    ff_dim: Optional[int] = None  # defaults to 4 * embed_dim
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self):
        if self.ff_dim is None:
            self.ff_dim = 4 * self.embed_dim
        assert self.embed_dim % self.n_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by n_heads ({self.n_heads})"
        )


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # optimization
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1

    # training schedule
    epochs: int = 10
    batch_size: int = 64
    eval_interval: int = 500
    log_interval: int = 100
    save_interval: int = 2000

    # data
    num_articles: int = 50000
    val_split: float = 0.05
    num_workers: int = 4

    # paths
    checkpoint_dir: str = "checkpoints"
    tokenizer_path: str = "tokenizer.json"

    # hardware
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    compile_model: bool = False  # torch.compile (PyTorch 2.0+)


# ── Preset model configurations ──────────────────────────────────────────────

MODEL_CONFIGS = {
    "small": ModelConfig(
        vocab_size=8192,
        context_length=256,
        n_layers=6,
        n_heads=6,
        embed_dim=384,
        dropout=0.1,
    ),
    "medium": ModelConfig(
        vocab_size=16384,
        context_length=512,
        n_layers=8,
        n_heads=8,
        embed_dim=512,
        dropout=0.1,
    ),
    "large": ModelConfig(
        vocab_size=32000,
        context_length=1024,
        n_layers=12,
        n_heads=12,
        embed_dim=768,
        dropout=0.1,
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """Return a preset model configuration by name."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Choose from: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[name]
