#!/usr/bin/env python3
"""
Setup training environment with compatible Unsloth + Gemma 4 dependencies.

This script ensures compatible versions are installed by:
1. Installing core dependencies first with exact versions
2. Then installing Unsloth (which will use existing versions)

Usage (from Colab or local):
    python scripts/setup_training_env.py
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=False,
        text=True
    )
    if result.returncode != 0:
        print(f"⚠️  Command failed with exit code {result.returncode}")
    return result.returncode == 0


def main():
    print("="*60)
    print("UNSLOOTH + GEMMA 4 TRAINING ENVIRONMENT SETUP")
    print("="*60)

    # Step 1: Upgrade pip
    run_command(
        "python -m pip install --upgrade pip",
        "Step 1: Upgrading pip"
    )

    # Step 2: Install core dependencies with exact versions
    # These are tested to work with Unsloth + Gemma 4
    core_packages = [
        ("transformers", "4.46.2"),
        ("trl", "0.9.6"),
        ("accelerate", "1.1.0"),
        ("bitsandbytes", "0.43.1"),
        ("datasets", "3.0.1"),
        ("peft", "0.12.0"),
    ]

    print("\n" + "="*60)
    print("Step 2: Installing core dependencies (exact versions)")
    print("="*60)

    for package, version in core_packages:
        success = run_command(
            f'pip install -q "{package}=={version}"',
            f"Installing {package}=={version}"
        )
        if not success:
            print(f"⚠️  Failed to install {package}, continuing...")

    # Step 3: Install Unsloth (will use already-installed versions)
    print("\n" + "="*60)
    print("Step 3: Installing Unsloth from GitHub")
    print("="*60)
    print("Using existing dependencies (not overriding)")

    success = run_command(
        'pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
        "Installing Unsloth"
    )

    # Step 4: Verify installation
    print("\n" + "="*60)
    print("Step 4: Verifying installation")
    print("="*60)

    packages_to_check = ["transformers", "trl", "accelerate", "unsloth"]
    for package in packages_to_check:
        result = subprocess.run(
            ["pip", "show", package],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            name = lines[0].split(': ')[1] if ': ' in lines[0] else package
            version = lines[1].split(': ')[1] if len(lines) > 1 and ': ' in lines[1] else "unknown"
            print(f"  ✅ {name}: {version}")
        else:
            print(f"  ❌ {package}: NOT FOUND")

    # Step 5: Test import
    print("\n" + "="*60)
    print("Step 5: Testing imports")
    print("="*60)

    try:
        from unsloth import FastModel
        print("  ✅ from unsloth import FastModel")
    except ImportError as e:
        print(f"  ❌ Failed to import unsloth.FastModel: {e}")
        print("\n🔧 Try restarting the runtime:")
        print("   Runtime → Restart session")
        sys.exit(1)

    try:
        from transformers import AutoTokenizer
        print("  ✅ from transformers import AutoTokenizer")
    except ImportError as e:
        print(f"  ❌ Failed to import transformers: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("✅ TRAINING ENVIRONMENT READY!")
    print("="*60)
    print("\nYou can now run training:")
    print("  python scripts/colab_train.py")


if __name__ == "__main__":
    main()
