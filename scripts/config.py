#!/usr/bin/env python3
"""
Configuration management for Gemma 4 Vietnamese Legal Documents project.
Loads settings from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_env_file(env_path: Optional[Path] = None) -> None:
    """
    Load environment variables from .env file.

    Args:
        env_path: Path to .env file. If None, uses project_root/.env
    """
    if env_path is None:
        env_path = get_project_root() / ".env"

    if not env_path.exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            # Only set if not already in environment
            if key not in os.environ:
                os.environ[key] = value


# Load .env on import
load_env_file()


@dataclass
class GitConfig:
    """Git repository configuration."""

    github_repo: str = field(default_factory=lambda: os.getenv(
        "GITHUB_REPO",
        "git@github.com:duyet/gemma-4-finetune-vietnamese-legal-documents.git"
    ))
    github_username: str = field(default_factory=lambda: os.getenv(
        "GITHUB_USERNAME", "duyet"
    ))
    hf_username: str = field(default_factory=lambda: os.getenv(
        "HF_USERNAME", "duyet"
    ))
    hf_repo_name: str = field(default_factory=lambda: os.getenv(
        "HF_REPO_NAME",
        "gemma-4-finetune-vietnamese-legal-documents"
    ))
    hf_dataset_name: str = field(default_factory=lambda: os.getenv(
        "HF_DATASET_NAME",
        "vietnamese-legal-documents"
    ))
    hf_model_name: str = field(default_factory=lambda: os.getenv(
        "HF_MODEL_NAME",
        "gemma-4-vietnamese-legal-rag"
    ))


@dataclass
class CrawlerConfig:
    """Web crawler configuration."""

    max_pages: int = field(default_factory=lambda: int(os.getenv(
        "CRAWLER_MAX_PAGES", "100"
    )))
    delay: float = field(default_factory=lambda: float(os.getenv(
        "CRAWLER_DELAY", "2.5"
    )))
    concurrent_requests: int = field(default_factory=lambda: int(os.getenv(
        "CRAWLER_CONCURRENT_REQUESTS", "4"
    )))
    user_agent: str = field(default_factory=lambda: os.getenv(
        "CRAWLER_USER_AGENT",
        "Mozilla/5.0 (compatible; TVPLBot/1.0; +https://github.com/duyet/gemma-4-finetune-vietnamese-legal-documents)"
    ))

    # Start URL
    start_url: str = field(default_factory=lambda: os.getenv(
        "CRAWLER_START_URL",
        "https://thuvienphapluat.vn/"
    ))


@dataclass
class TrainingConfig:
    """Model training configuration."""

    # Model settings
    base_model: str = field(default_factory=lambda: os.getenv(
        "BASE_MODEL",
        "unsloth/gemma-4-E2B-it"
    ))
    max_seq_length: int = field(default_factory=lambda: int(os.getenv(
        "MAX_SEQ_LENGTH", "4096"
    )))
    load_in_4bit: bool = field(default_factory=lambda: os.getenv(
        "LOAD_IN_4BIT", "true"
    ).lower() == "true")

    # LoRA settings
    lora_r: int = field(default_factory=lambda: int(os.getenv(
        "LORA_R", "16"
    )))
    lora_alpha: int = field(default_factory=lambda: int(os.getenv(
        "LORA_ALPHA", "16"
    )))
    lora_dropout: float = field(default_factory=lambda: float(os.getenv(
        "LORA_DROPOUT", "0.05"
    )))

    # Training hyperparameters
    batch_size: int = field(default_factory=lambda: int(os.getenv(
        "BATCH_SIZE", "2"
    )))
    gradient_accumulation_steps: int = field(default_factory=lambda: int(os.getenv(
        "GRADIENT_ACCUMULATION_STEPS", "4"
    )))
    learning_rate: float = field(default_factory=lambda: float(os.getenv(
        "LEARNING_RATE", "2e-4"
    )))
    num_epochs: int = field(default_factory=lambda: int(os.getenv(
        "NUM_EPOCHS", "1"
    )))

    # Output paths
    output_dir: str = field(default_factory=lambda: os.getenv(
        "OUTPUT_DIR",
        "outputs_pretrain"
    ))


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""

    embedding_model: str = field(default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL",
        "bkai-foundation-models/vietnamese-bi-encoder"
    ))
    vector_store_path: str = field(default_factory=lambda: os.getenv(
        "VECTOR_STORE_PATH",
        "./data/chroma_db"
    ))
    top_k_retrieval: int = field(default_factory=lambda: int(os.getenv(
        "TOP_K_RETRIEVAL", "3"
    )))

    # Generation settings
    temperature: float = field(default_factory=lambda: float(os.getenv(
        "TEMPERATURE", "0.7"
    )))
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv(
        "MAX_NEW_TOKENS", "512"
    )))


@dataclass
class PathConfig:
    """Project path configuration."""

    project_root: Path = field(default_factory=get_project_root)
    data_dir: Path = field(default_factory=lambda: get_project_root() / "data")
    raw_data_dir: Path = field(default_factory=lambda: get_project_root() / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: get_project_root() / "data" / "processed")
    pretrain_dir: Path = field(default_factory=lambda: get_project_root() / "data" / "pretrain")
    sft_dir: Path = field(default_factory=lambda: get_project_root() / "data" / "sft")
    hf_dataset_dir: Path = field(default_factory=lambda: get_project_root() / "data" / "hf_dataset")


# Global config instances
git = GitConfig()
crawler = CrawlerConfig()
training = TrainingConfig()
rag = RAGConfig()
paths = PathConfig()


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("Current Configuration")
    print("=" * 60)
    print()
    print("[Git Configuration]")
    print(f"  GitHub Repo: {git.github_repo}")
    print(f"  GitHub Username: {git.github_username}")
    print(f"  HF Username: {git.hf_username}")
    print(f"  HF Dataset: {git.hf_username}/{git.hf_dataset_name}")
    print(f"  HF Model: {git.hf_username}/{git.hf_model_name}")
    print()
    print("[Crawler Configuration]")
    print(f"  Max Pages: {crawler.max_pages}")
    print(f"  Delay: {crawler.delay}s")
    print(f"  Concurrent: {crawler.concurrent_requests}")
    print()
    print("[Training Configuration]")
    print(f"  Base Model: {training.base_model}")
    print(f"  Max Seq Length: {training.max_seq_length}")
    print(f"  Batch Size: {training.batch_size}")
    print(f"  Learning Rate: {training.learning_rate}")
    print(f"  LoRA R: {training.lora_r}")
    print()
    print("[RAG Configuration]")
    print(f"  Embedding Model: {rag.embedding_model}")
    print(f"  Vector Store: {rag.vector_store_path}")
    print(f"  Top K: {rag.top_k_retrieval}")
    print()
    print("[Path Configuration]")
    print(f"  Project Root: {paths.project_root}")
    print(f"  Data Dir: {paths.data_dir}")
    print()


if __name__ == "__main__":
    print_config()
