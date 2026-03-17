# WikiLM — A Language Model Trained on Wikipedia

A from-scratch GPT-style transformer language model trained on Wikipedia text data. Built entirely in PyTorch with no dependency on pre-trained weights — every component from the BPE tokenizer to the transformer decoder is implemented and trained from the ground up.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

WikiLM is an educational and practical project that demonstrates how to build, train, and deploy a language model end-to-end. The model learns to generate coherent English text by training on Wikipedia articles.

**Key highlights:**
- Custom GPT architecture (transformer decoder) implemented from scratch in PyTorch
- Byte-Pair Encoding (BPE) tokenizer trained on the corpus
- Wikipedia data pipeline using HuggingFace `datasets`
- Configurable model sizes (small / medium / large)
- Text generation with top-k, top-p (nucleus), and temperature sampling
- Interactive web demo via Gradio

## Architecture

```
WikiLM (Transformer Decoder)
├── Token Embedding + Positional Embedding
├── N x Transformer Blocks
│   ├── Multi-Head Causal Self-Attention
│   ├── Layer Normalization (Pre-Norm)
│   └── Feed-Forward Network (GELU activation)
├── Final Layer Norm
└── Linear Head → Vocabulary
```

### Model Configurations

| Config | Layers | Heads | Embed Dim | Params  | Context |
|--------|--------|-------|-----------|---------|---------|
| Small  | 6      | 6     | 384       | ~25M    | 256     |
| Medium | 8      | 8     | 512       | ~60M    | 512     |
| Large  | 12     | 12    | 768       | ~124M   | 1024    |

## Quick Start

### 1. Installation

```bash
git clone https://github.com/getmokshshah/wiki-lm.git
cd wiki-lm
pip install -r requirements.txt
```

### 2. Train the Tokenizer

```bash
python tokenizer.py --vocab_size 8192 --num_articles 50000
```

This downloads Wikipedia articles and trains a BPE tokenizer on the corpus.

### 3. Train the Model

```bash
# Train the small model (recommended for getting started)
python train.py --config small --epochs 10 --batch_size 64

# Train the medium model (needs a GPU with 12GB+ VRAM)
python train.py --config medium --epochs 15 --batch_size 32

# Resume from a checkpoint
python train.py --config small --resume checkpoints/wikilm_small_epoch5.pt
```

### 4. Generate Text

```bash
python generate.py \
  --checkpoint checkpoints/wikilm_small_best.pt \
  --prompt "The history of artificial intelligence" \
  --max_tokens 200 \
  --temperature 0.8 \
  --top_k 50
```

### 5. Launch the Web Demo

```bash
python app.py
```

This starts a Gradio interface at `http://localhost:7860` where you can interactively generate text.

## Project Structure

```
wiki-lm/
├── config.py          # Model and training hyperparameters
├── model.py           # GPT transformer architecture
├── dataset.py         # Wikipedia data loading and preprocessing
├── tokenizer.py       # BPE tokenizer (train & encode/decode)
├── train.py           # Training loop with logging and checkpoints
├── generate.py        # Text generation with sampling strategies
├── app.py             # Gradio web interface
├── requirements.txt   # Python dependencies
└── README.md
```

## Training Details

**Data pipeline:**
1. Download English Wikipedia articles via HuggingFace `datasets`
2. Clean and filter articles (remove stubs, metadata, markup)
3. Train a BPE tokenizer on the cleaned corpus
4. Tokenize articles and chunk into fixed-length sequences
5. Serve batches with a PyTorch `DataLoader`

**Training recipe:**
- Optimizer: AdamW (β₁=0.9, β₂=0.95, weight decay=0.1)
- Learning rate: cosine annealing with linear warmup (first 10% of steps)
- Gradient clipping: max norm 1.0
- Mixed precision: FP16 via `torch.cuda.amp` (when GPU available)
- Checkpointing: saves best model (by validation loss) and periodic snapshots

## Web Demo

The Gradio app (`app.py`) provides an interactive interface where users can:
- Type a prompt and generate text completions
- Adjust generation parameters (temperature, top-k, top-p, max length)
- See token-by-token streaming output

To deploy on Hugging Face Spaces:
1. Create a new Space (Gradio SDK)
2. Upload the model checkpoint, tokenizer files, and `app.py`
3. The app auto-launches on the Space

## Results

After training the small model for 10 epochs on ~50K Wikipedia articles:
- **Training loss**: ~3.2 (cross-entropy)
- **Validation loss**: ~3.5
- The model learns coherent English grammar, factual patterns, and Wikipedia-style writing

Sample generation (temperature=0.8):
> **Prompt:** "The theory of relativity"
> **Output:** "The theory of relativity is a fundamental framework in modern physics developed by Albert Einstein in the early twentieth century. It describes the relationship between space and time, and how gravity affects the curvature of spacetime..."

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) for architectural inspiration
- HuggingFace `datasets` for the Wikipedia data pipeline
- The PyTorch team for an excellent deep learning framework
