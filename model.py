"""
WikiLM Model
=============
A GPT-style transformer decoder implemented from scratch in PyTorch.
Pre-norm architecture with causal multi-head self-attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.embed_dim % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.embed_dim = config.embed_dim

        # key, query, value projections combined
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=config.bias)
        # output projection
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask: lower-triangular
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        # reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # apply attention to values
        out = attn @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.ff_dim, bias=config.bias)
        self.fc2 = nn.Linear(config.ff_dim, config.embed_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class WikiLM(nn.Module):
    """
    WikiLM: A GPT-style transformer decoder language model.

    Architecture:
        Token embeddings + learned positional embeddings
        → N transformer blocks (pre-norm, causal attention)
        → Layer norm → linear head (weight-tied with token embeddings)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # weight tying: share token embedding weights with the output head
        self.head.weight = self.token_emb.weight

        # initialize weights
        self.apply(self._init_weights)
        # apply scaled init to residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Args:
            idx: Token indices, shape (B, T)
            targets: Target indices for loss computation, shape (B, T)

        Returns:
            logits: Shape (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.size()
        assert T <= self.config.context_length, (
            f"Sequence length {T} exceeds context_length {self.config.context_length}"
        )

        # embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_emb(idx)       # (B, T, embed_dim)
        pos_emb = self.pos_emb(pos)          # (T, embed_dim)
        x = self.drop(tok_emb + pos_emb)

        # transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with sampling.

        Args:
            idx: Starting token indices, shape (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Keep only top-k tokens (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)

        Returns:
            Token indices including generated tokens, shape (B, T + max_new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # crop to context length
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "WikiLM":
        """Load a model from a checkpoint file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model
