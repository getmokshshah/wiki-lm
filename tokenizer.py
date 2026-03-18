"""
WikiLM Tokenizer
================
Byte-Pair Encoding (BPE) tokenizer trained on Wikipedia text.
Supports training from a corpus, saving/loading, and encoding/decoding text.
"""

import json
import re
import argparse
from collections import Counter, defaultdict
from typing import Optional

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class BPETokenizer:
    """
    A Byte-Pair Encoding tokenizer.

    Trains merge rules from a text corpus and uses them to tokenize
    new text into subword units.
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self.vocab_size: int = 0

    def train(self, texts: list[str], vocab_size: int = 8192, min_frequency: int = 2):
        """
        Train BPE merges from a list of texts.

        Args:
            texts: List of training text strings
            vocab_size: Target vocabulary size (including special tokens)
            min_frequency: Minimum pair frequency to consider for merging
        """
        print(f"Training BPE tokenizer (target vocab_size={vocab_size})...")

        # Step 1: Build initial character-level vocabulary from word frequencies
        word_freqs = Counter()
        for text in texts:
            words = re.findall(r"\w+|[^\w\s]", text.lower())
            for word in words:
                # represent each word as space-separated characters + end-of-word marker
                chars = " ".join(list(word)) + " </w>"
                word_freqs[chars] += 1

        # Step 2: Build initial character vocab
        char_vocab = set()
        for word in word_freqs:
            for ch in word.split():
                char_vocab.add(ch)

        # reserve space for special tokens
        num_merges = vocab_size - len(char_vocab) - len(SPECIAL_TOKENS)
        if num_merges <= 0:
            print(f"Warning: vocab_size too small for corpus. Need at least {len(char_vocab) + len(SPECIAL_TOKENS) + 1}")
            num_merges = 100

        # Step 3: Iteratively find and apply the most frequent pair
        merges = []
        for i in range(num_merges):
            pairs = self._get_pair_frequencies(word_freqs)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_frequency:
                break

            merges.append(best_pair)
            word_freqs = self._apply_merge(word_freqs, best_pair)

            if (i + 1) % 500 == 0:
                print(f"  Merge {i + 1}/{num_merges} — merged '{best_pair[0]}' + '{best_pair[1]}'")

        # Step 4: Build final vocabulary
        self.merges = merges
        self._build_vocab(word_freqs)
        print(f"Tokenizer trained: {self.vocab_size} tokens, {len(self.merges)} merges")

    def _get_pair_frequencies(self, word_freqs: dict[str, int]) -> dict[tuple[str, str], int]:
        """Count frequencies of adjacent symbol pairs across all words."""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _apply_merge(
        self, word_freqs: dict[str, int], pair: tuple[str, str]
    ) -> dict[str, int]:
        """Merge all occurrences of a pair in the vocabulary."""
        new_freqs = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        replacement = "".join(pair)

        for word, freq in word_freqs.items():
            new_word = pattern.sub(replacement, word)
            new_freqs[new_word] = freq

        return new_freqs

    def _build_vocab(self, word_freqs: dict[str, int]):
        """Build the token-to-id vocabulary from merged word frequencies."""
        tokens = set()
        for word in word_freqs:
            for token in word.split():
                tokens.add(token)

        # build ordered vocab: special tokens first, then sorted tokens
        self.vocab = {}
        for i, st in enumerate(SPECIAL_TOKENS):
            self.vocab[st] = i

        for i, token in enumerate(sorted(tokens), start=len(SPECIAL_TOKENS)):
            if token not in self.vocab:
                self.vocab[token] = i

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode a text string into a list of token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to prepend BOS and append EOS

        Returns:
            List of integer token IDs
        """
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.vocab[BOS_TOKEN])

        for word in words:
            symbols = list(word) + ["</w>"]
            # apply merges in order
            for merge_pair in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == merge_pair[0] and symbols[i + 1] == merge_pair[1]:
                        symbols[i] = merge_pair[0] + merge_pair[1]
                        del symbols[i + 1]
                    else:
                        i += 1

            for sym in symbols:
                token_ids.append(self.vocab.get(sym, self.vocab[UNK_TOKEN]))

        if add_special_tokens:
            token_ids.append(self.vocab[EOS_TOKEN])

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Args:
            token_ids: List of integer token IDs

        Returns:
            Decoded text string
        """
        tokens = []
        for tid in token_ids:
            token = self.inverse_vocab.get(tid, UNK_TOKEN)
            if token in SPECIAL_TOKENS:
                continue
            tokens.append(token)

        # join tokens and clean up end-of-word markers
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def save(self, path: str):
        """Save tokenizer to a JSON file."""
        data = {
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "vocab_size": self.vocab_size,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.vocab = data["vocab"]
        tokenizer.inverse_vocab = {int(v): k for k, v in data["vocab"].items()}
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.vocab_size = data["vocab_size"]
        return tokenizer

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.vocab[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.vocab[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.vocab[EOS_TOKEN]


def fetch_wikipedia_texts(num_articles: int = 50000) -> list[str]:
    """Download Wikipedia articles using HuggingFace datasets."""
    from datasets import load_dataset

    print(f"Downloading {num_articles} Wikipedia articles...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    texts = []
    for i, article in enumerate(dataset):
        if i >= num_articles:
            break
        text = article["text"].strip()
        if len(text) > 200:  # skip stubs
            texts.append(text[:5000])  # cap article length for tokenizer training
        if (i + 1) % 10000 == 0:
            print(f"  Fetched {i + 1} articles...")

    print(f"Collected {len(texts)} articles")
    return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on Wikipedia")
    parser.add_argument("--vocab_size", type=int, default=8192, help="Target vocabulary size")
    parser.add_argument("--num_articles", type=int, default=50000, help="Number of Wikipedia articles to use")
    parser.add_argument("--output", type=str, default="tokenizer.json", help="Output path")
    args = parser.parse_args()

    texts = fetch_wikipedia_texts(args.num_articles)
    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=args.vocab_size)
    tokenizer.save(args.output)

    # quick sanity check
    test = "The history of artificial intelligence began in antiquity."
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded)
    print(f"\nSanity check:")
    print(f"  Original: {test}")
    print(f"  Encoded:  {encoded[:20]}...")
    print(f"  Decoded:  {decoded}")
