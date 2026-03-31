"""
============================================================
PRODIGY INFOTECH – Generative AI Internship
Task-01: Text Generation with GPT-2
============================================================
Description:
    Fine-tune / use a pre-trained GPT-2 model to generate
    coherent and contextually relevant text based on a given
    prompt. Demonstrates text generation using HuggingFace
    Transformers library.

Requirements:
    pip install transformers torch

Usage:
    python task01_text_generation_gpt2.py
    python task01_text_generation_gpt2.py --prompt "Once upon a time" --max_length 200
============================================================
"""

import argparse
import os
import sys

# ── Check dependencies ────────────────────────────────────
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
except ImportError:
    print("[ERROR] Missing dependencies. Install them with:")
    print("        pip install transformers torch")
    sys.exit(1)


# ── Configuration ─────────────────────────────────────────
DEFAULT_PROMPT     = "The future of artificial intelligence is"
DEFAULT_MAX_LENGTH = 250
DEFAULT_NUM_SEQS   = 3
MODEL_NAME         = "gpt2"          # Options: gpt2, gpt2-medium, gpt2-large
OUTPUT_FILE        = "task01_output.txt"


def load_model(model_name: str):
    """Load GPT-2 tokenizer and model from HuggingFace Hub."""
    print(f"\n[INFO] Loading GPT-2 model ({model_name})...")
    print("       (This may take a moment on first run – model is downloaded & cached)")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model     = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[INFO] Model loaded. Running on: {device.upper()}")
    return tokenizer, model, device


def generate_text(
    prompt:      str,
    tokenizer,
    model,
    device:      str,
    max_length:  int = DEFAULT_MAX_LENGTH,
    num_seqs:    int = DEFAULT_NUM_SEQS,
    temperature: float = 0.9,
    top_k:       int   = 50,
    top_p:       float = 0.95,
) -> list[str]:
    """
    Generate text continuations for the given prompt.

    Args:
        prompt      : Starting text for generation
        tokenizer   : GPT-2 tokenizer
        model       : GPT-2 model
        device      : 'cpu' or 'cuda'
        max_length  : Maximum total tokens in output
        num_seqs    : Number of distinct sequences to generate
        temperature : Controls randomness (lower = more deterministic)
        top_k       : Keep only top-k tokens at each step
        top_p       : Nucleus sampling probability threshold

    Returns:
        List of generated text strings
    """
    tokenizer.pad_token = tokenizer.eos_token  # avoid warning
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length          = max_length,
            num_return_sequences= num_seqs,
            do_sample           = True,
            temperature         = temperature,
            top_k               = top_k,
            top_p               = top_p,
            pad_token_id        = tokenizer.eos_token_id,
            repetition_penalty  = 1.2,
        )

    # Decode and strip the original prompt from each output
    results = []
    for output in outputs:
        full_text = tokenizer.decode(output, skip_special_tokens=True)
        results.append(full_text)
    return results


def save_output(prompt: str, results: list[str], filepath: str):
    """Save generated texts to a plain-text file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("PRODIGY INFOTECH – Task-01: GPT-2 Text Generation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"PROMPT:\n  {prompt}\n\n")
        for i, text in enumerate(results, 1):
            f.write(f"{'─' * 60}\n")
            f.write(f"[Generated Sequence {i}]\n")
            f.write(f"{text}\n\n")
    print(f"\n[INFO] Output saved to: {os.path.abspath(filepath)}")


def print_results(prompt: str, results: list[str]):
    """Pretty-print results to the terminal."""
    print("\n" + "=" * 60)
    print("Generated Text")
    print("=" * 60)
    print(f"PROMPT: {prompt}\n")
    for i, text in enumerate(results, 1):
        print(f"{'─' * 60}")
        print(f"[Sequence {i}]")
        print(text)
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task-01: Text Generation with GPT-2"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Input prompt for text generation (default: '{DEFAULT_PROMPT}')"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Maximum token length of generated text (default: {DEFAULT_MAX_LENGTH})"
    )
    parser.add_argument(
        "--num_seqs",
        type=int,
        default=DEFAULT_NUM_SEQS,
        help=f"Number of sequences to generate (default: {DEFAULT_NUM_SEQS})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (default: 0.9)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        choices=["gpt2", "gpt2-medium", "gpt2-large"],
        help="GPT-2 model variant to use (default: gpt2)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("PRODIGY INFOTECH – Generative AI Internship")
    print("Task-01: Text Generation with GPT-2")
    print("=" * 60)
    print(f"  Prompt      : {args.prompt}")
    print(f"  Max Length  : {args.max_length} tokens")
    print(f"  Sequences   : {args.num_seqs}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Model       : {args.model}")

    # Load model
    tokenizer, model, device = load_model(args.model)

    print(f"\n[INFO] Generating {args.num_seqs} text sequence(s)...")
    results = generate_text(
        prompt      = args.prompt,
        tokenizer   = tokenizer,
        model       = model,
        device      = device,
        max_length  = args.max_length,
        num_seqs    = args.num_seqs,
        temperature = args.temperature,
    )

    # Display and save
    print_results(args.prompt, results)
    save_output(args.prompt, results, OUTPUT_FILE)
    print("\n[DONE] Task-01 complete!")


if __name__ == "__main__":
    main()
