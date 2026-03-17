"""
WikiLM Dataset
==============
Wikipedia data loading, preprocessing, and PyTorch Dataset/DataLoader creation.
"""

import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from tokenizer import BPETokenizer


class WikiTextDataset(Dataset):
    """
    A PyTorch Dataset that serves fixed-length token sequences from
    a pre-tokenized Wikipedia corpus.

    Each sample is a (input, target) pair where target is input shifted by one token.
    """

    def __init__(self, token_ids: list[int], context_length: int):
        """
        Args:
            token_ids: Flat list of token IDs from the entire corpus
            context_length: Number of tokens per training sequence
        """
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

        # number of complete sequences we can form
        self.n_sequences = (len(self.token_ids) - 1) // context_length

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.context_length
        end = start + self.context_length

        x = self.token_ids[start:end]
        y = self.token_ids[start + 1 : end + 1]

        return x, y


def clean_article(text: str) -> str:
    """Clean a Wikipedia article by removing markup, references, and metadata."""
    # remove section headers that are just formatting
    text = re.sub(r"={2,}.*?={2,}", "", text)
    # remove reference markers like [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", text)
    # remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # remove extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def load_wikipedia_articles(num_articles: int = 50000, min_length: int = 300) -> list[str]:
    """
    Load and clean Wikipedia articles using HuggingFace datasets.

    Args:
        num_articles: Number of articles to load
        min_length: Minimum article length in characters (skip stubs)

    Returns:
        List of cleaned article strings
    """
    from datasets import load_dataset

    print(f"Loading {num_articles} Wikipedia articles...")
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    articles = []
    seen = 0
    for article in dataset:
        if seen >= num_articles:
            break
        seen += 1

        text = clean_article(article["text"])
        if len(text) >= min_length:
            articles.append(text)

        if seen % 10000 == 0:
            print(f"  Processed {seen} articles, kept {len(articles)}")

    print(f"Loaded {len(articles)} articles (filtered from {seen})")
    return articles


def tokenize_corpus(articles: list[str], tokenizer: BPETokenizer) -> list[int]:
    """
    Tokenize an entire corpus into a flat list of token IDs.

    Each article is bounded by BOS/EOS tokens.
    """
    print("Tokenizing corpus...")
    all_ids = []
    for i, article in enumerate(articles):
        ids = tokenizer.encode(article, add_special_tokens=True)
        all_ids.extend(ids)
        if (i + 1) % 10000 == 0:
            print(f"  Tokenized {i + 1}/{len(articles)} articles ({len(all_ids):,} tokens)")

    print(f"Total tokens: {len(all_ids):,}")
    return all_ids


def create_dataloaders(
    token_ids: list[int],
    context_length: int,
    batch_size: int,
    val_split: float = 0.05,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders from tokenized data.

    Args:
        token_ids: Flat list of token IDs
        context_length: Sequence length for training
        batch_size: Batch size
        val_split: Fraction of data for validation
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    dataset = WikiTextDataset(token_ids, context_length)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"DataLoaders created: {train_size} train sequences, {val_size} val sequences")
    print(f"  Batches per epoch: {len(train_loader)} train, {len(val_loader)} val")

    return train_loader, val_loader
