"""
WikiLM Web Demo
===============
Interactive Gradio interface for generating text with a trained WikiLM model.
Deploy locally or on Hugging Face Spaces.
"""

import os
import argparse
import torch
import gradio as gr

from model import WikiLM
from tokenizer import BPETokenizer
from generate import generate_text


# ── Global model references ──────────────────────────────────────────────────
MODEL: WikiLM = None
TOKENIZER: BPETokenizer = None
DEVICE: str = "cpu"


def load_model(checkpoint_path: str, tokenizer_path: str = "tokenizer.json"):
    """Load model and tokenizer into global state."""
    global MODEL, TOKENIZER, DEVICE

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    print(f"Loading model on {DEVICE}...")
    MODEL = WikiLM.from_checkpoint(checkpoint_path, device=DEVICE)
    TOKENIZER = BPETokenizer.load(tokenizer_path)
    print(f"Model loaded: {MODEL.count_parameters():,} parameters")


def predict(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    """Generate text from user input."""
    if MODEL is None:
        return "⚠️ Model not loaded. Please provide a valid checkpoint."

    if not prompt.strip():
        return "Please enter a prompt to get started."

    try:
        output = generate_text(
            model=MODEL,
            tokenizer=TOKENIZER,
            prompt=prompt.strip(),
            max_new_tokens=int(max_tokens),
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            device=DEVICE,
        )
        return output
    except Exception as e:
        return f"Generation error: {str(e)}"


# ── Gradio Interface ─────────────────────────────────────────────────────────

DESCRIPTION = """
# 🧠 WikiLM — Text Generation

A GPT-style language model trained from scratch on Wikipedia text data.
Type a prompt and watch the model complete it.

**How it works:** The model was trained on ~50K Wikipedia articles using a custom
BPE tokenizer and transformer decoder architecture. It generates text one token at a
time using autoregressive sampling.
"""

EXAMPLES = [
    ["The history of artificial intelligence", 200, 0.8, 50, 0.9],
    ["In quantum mechanics, the wave function", 150, 0.7, 40, 0.85],
    ["The United States Constitution was", 200, 0.8, 50, 0.9],
    ["Machine learning is a subset of", 180, 0.75, 50, 0.9],
    ["The theory of evolution by natural selection", 200, 0.8, 50, 0.9],
]


def build_interface() -> gr.Blocks:
    """Build the Gradio Blocks interface."""
    with gr.Blocks(
        title="WikiLM — Text Generation",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.teal,
            neutral_hue=gr.themes.colors.slate,
        ),
        css="""
        .output-text { font-size: 1.05rem; line-height: 1.8; }
        footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your text prompt here...",
                    lines=3,
                    max_lines=6,
                )
                output = gr.Textbox(
                    label="Generated Text",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                    elem_classes=["output-text"],
                )
                generate_btn = gr.Button("Generate ✨", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Generation Settings")
                max_tokens = gr.Slider(
                    minimum=50, maximum=500, value=200, step=10,
                    label="Max Tokens",
                    info="Number of tokens to generate",
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.8, step=0.05,
                    label="Temperature",
                    info="Lower = more focused, higher = more creative",
                )
                top_k = gr.Slider(
                    minimum=0, maximum=200, value=50, step=5,
                    label="Top-K",
                    info="Keep only top-K tokens (0 = disabled)",
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                    label="Top-P (Nucleus)",
                    info="Cumulative probability cutoff",
                )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[prompt, max_tokens, temperature, top_k, top_p],
            label="Example Prompts",
        )

        generate_btn.click(
            fn=predict,
            inputs=[prompt, max_tokens, temperature, top_k, top_p],
            outputs=output,
        )
        prompt.submit(
            fn=predict,
            inputs=[prompt, max_tokens, temperature, top_k, top_p],
            outputs=output,
        )

        gr.Markdown(
            "---\n"
            "Built by [Moksh Shah](https://mokshshah.dev) · "
            "[GitHub](https://github.com/getmokshshah/wiki-lm) · "
            "Architecture: Transformer Decoder (GPT-style) · "
            "Trained on Wikipedia"
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WikiLM Web Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get("WIKILM_CHECKPOINT", "checkpoints/wikilm_small_best.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=os.environ.get("WIKILM_TOKENIZER", "tokenizer.json"),
        help="Path to tokenizer",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    args = parser.parse_args()

    load_model(args.checkpoint, args.tokenizer)

    demo = build_interface()
    demo.launch(server_port=args.port, share=args.share)
