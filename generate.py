"""
WikiLM Text Generation
======================
Generate text from a trained WikiLM checkpoint using various sampling strategies.
"""

import argparse
import torch

from model import WikiLM
from tokenizer import BPETokenizer


def generate_text(
    model: WikiLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cpu",
) -> str:
    """
    Generate text from a prompt.

    Args:
        model: Trained WikiLM model
        tokenizer: BPE tokenizer
        prompt: Starting text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        device: Device to run on

    Returns:
        Generated text string
    """
    model.eval()

    # encode prompt
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    # generate
    output_ids = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # decode
    generated = tokenizer.decode(output_ids[0].tolist())
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text with WikiLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--prompt", type=str, default="The history of", help="Starting prompt")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    args = parser.parse_args()

    # device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Loading model from {args.checkpoint}...")
    model = WikiLM.from_checkpoint(args.checkpoint, device=device)
    print(f"  Parameters: {model.count_parameters():,}")

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"  Vocabulary: {tokenizer.vocab_size} tokens")

    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("=" * 60)

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n── Sample {i + 1} ──")

        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        print(text)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
