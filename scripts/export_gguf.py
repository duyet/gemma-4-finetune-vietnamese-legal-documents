#!/usr/bin/env python3
"""
Export trained model to GGUF format for local inference.

Usage:
    python scripts/export_gguf.py --model-dir gemma4_e2b_tvpl_pretrain_lora --quantization q4_k_m
"""

import argparse
import json
import sys
from pathlib import Path

from unsloth import FastModel


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to GGUF format")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory containing LoRA adapters (auto-detected if not set)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for GGUF files (auto-generated from model-dir if not set)")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="Quantization method")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Maximum sequence length")
    return parser.parse_args()


def auto_detect_model_dir():
    """Auto-detect the latest model directory."""
    import glob

    # Look for directories matching the pattern
    pattern = "gemma-*-vi-legal-pretrain"
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"No model directory found matching pattern '{pattern}'\n"
            f"Please run training first or specify --model-dir"
        )

    # Sort by modification time, get the latest
    latest = max(matches, key=lambda p: Path(p).stat().st_mtime)
    print(f"🔍 Auto-detected model directory: {latest}")
    return latest


def get_output_dir(args) -> str:
    """Generate output directory name from model directory."""
    if args.output_dir:
        return args.output_dir

    # gemma-4-E2B-it-vi-legal-pretrain -> gemma-4-E2B-it-vi-legal-gguf
    model_dir_name = Path(args.model_dir).name
    return model_dir_name.replace("-pretrain", "-gguf")


def export_to_gguf(args):
    """Export model to GGUF format."""
    print("\n" + "="*60)
    print("EXPORTING TO GGUF")
    print("="*60)
    print(f"Model: {args.model_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Quantization: {args.quantization}")
    print("="*60)

    # Check if model directory exists
    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"\n❌ Error: Model directory not found: {args.model_dir}")
        print("\nPlease train the model first:")
        print("  python scripts/colab_train.py")
        sys.exit(1)

    # Load model
    print(f"\n🤖 Loading model from {args.model_dir}")
    try:
        model, tokenizer = FastModel.from_pretrained(
            args.model_dir,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        print("✅ Model loaded")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        sys.exit(1)

    # Export
    print(f"\n📦 Exporting to GGUF format ({args.quantization})...")
    print("   This may take 10-30 minutes...")

    try:
        model.save_pretrained_gguf(
            args.output_dir,
            tokenizer,
            quantization_method=args.quantization,
        )
        print("✅ Export complete")
    except Exception as e:
        print(f"\n❌ Error exporting: {e}")
        sys.exit(1)

    # Show output files
    output_path = Path(args.output_dir)
    if output_path.exists():
        gguf_files = list(output_path.glob("*.gguf"))
        print(f"\n📁 Generated files:")
        for f in gguf_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   {f.name} ({size_mb:.1f} MB)")

    # Save export info
    info = {
        "source_model": args.model_dir,
        "output_dir": args.output_dir,
        "quantization": args.quantization,
        "max_seq_length": args.max_seq_length,
        "files": [f.name for f in gguf_files],
    }

    info_path = output_path / "export_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Export info saved to {info_path}")
    print("\nYou can now use the model with llama.cpp:")
    print(f"  ./main -m {args.output_dir}/*.gguf -p \"Your question here\"")


def main():
    args = parse_args()

    # Auto-detect model directory if not specified
    if args.model_dir is None:
        args.model_dir = auto_detect_model_dir()

    args.output_dir = get_output_dir(args)
    export_to_gguf(args)


if __name__ == "__main__":
    main()
