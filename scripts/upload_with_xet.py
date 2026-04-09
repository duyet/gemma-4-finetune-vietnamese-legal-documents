#!/usr/bin/env python3
"""
Upload large datasets to HuggingFace using XET (experimental transfer).

XET is faster for large files (>500MB). Install:
pip install huggingface_hub[xet]

Usage:
    uv run python scripts/upload_with_xet.py -r username/dataset-name -d data/hf_dataset
"""

import sys
from pathlib import Path

import click


@click.command()
@click.option("--repo-id", "-r", required=True, help="Target repo ID (e.g., username/dataset-name)")
@click.option("--data-dir", "-d", default="data/hf_dataset", help="Local dataset directory")
@click.option("--private", is_flag=True, help="Make repository private")
@click.option("--use-xet", is_flag=True, default=True, help="Use XET for faster transfer")
def main(repo_id: str, data_dir: str, private: bool, use_xet: bool):
    """Upload dataset to HuggingFace Hub using XET if available."""
    from huggingface_hub import HfApi, login

    print("=" * 60)
    print("HuggingFace Upload with XET")
    print("=" * 60)

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        return 1

    # Check XET availability
    if use_xet:
        try:
            import huggingface_hub.xet as xet_module
            print("✅ XET is available - will use for large files")
        except ImportError:
            print("⚠️  XET not available, install with: pip install huggingface_hub[xet]")
            print("   Falling back to regular upload...")
            use_xet = False

    # Login
    try:
        from huggingface_hub import whoami
        whoami()
        print(f"✅ Logged in as: {whoami()['name']}")
    except Exception:
        print("Please login to HuggingFace:")
        from huggingface_hub import login
        login()
        print(f"✅ Logged in")

    api = HfApi()

    # Create repository
    print(f"\n[1/3] Creating repository: {repo_id}")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        print(f"✅ Repository ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"⚠️  Repository creation note: {e}")

    # Upload using XET if available
    if use_xet:
        print(f"\n[2/3] Uploading with XET (fast for large files)...")
        try:
            # Use XET's streaming upload
            from huggingface_hub.xet import upload_folder

            upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(data_path),
            )
            print(f"✅ Upload complete with XET!")
        except Exception as e:
            print(f"⚠️  XET upload failed: {e}")
            print("   Falling back to regular upload...")
            use_xet = False

    # Regular upload as fallback
    if not use_xet:
        print(f"\n[2/3] Uploading files...")
        try:
            # Upload README first
            readme_path = data_path / "README.md"
            if readme_path.exists():
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_content = f.read()
                api.upload_file(
                    path_or_fileobj=readme_content.encode(),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(f"  ✅ README.md")

            # Upload each config folder
            configs = [d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            for config_dir in configs:
                config_name = config_dir.name
                print(f"  Uploading {config_name}/...")

                # Upload dataset_info.json
                info_file = config_dir / "dataset_info.json"
                if info_file.exists():
                    api.upload_file(
                        path_or_fileobj=str(info_file),
                        path_in_repo=f"{config_name}/dataset_info.json",
                        repo_id=repo_id,
                        repo_type="dataset",
                    )

                # Upload train folder
                train_dir = config_dir / "train"
                if train_dir.exists():
                    # Upload all files in train folder
                    for file_path in train_dir.iterdir():
                        if file_path.is_file():
                            api.upload_file(
                                path_or_fileobj=str(file_path),
                                path_in_repo=f"{config_name}/train/{file_path.name}",
                                repo_id=repo_id,
                                repo_type="dataset",
                            )
                print(f"    ✅ {config_name}")

        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return 1

    # Summary
    print(f"\n[3/3] Upload Complete!")
    print(f"\n📍 View at: https://huggingface.co/datasets/{repo_id}")
    print(f"\nTo use in Python:")
    print(f"  from datasets import load_dataset")
    print(f"  docs = load_dataset('{repo_id}', 'documents')")
    print(f"  passages = load_dataset('{repo_id}', 'passages')")
    print(f"  sft = load_dataset('{repo_id}', 'sft')")

    return 0


if __name__ == "__main__":
    sys.exit(main())
