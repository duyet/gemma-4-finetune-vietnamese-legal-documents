# HuggingFace XET Upload Guide

## What is XET?

XET (Experimental Transfer) is HuggingFace's optimized transfer protocol for large files (>500MB). It's significantly faster than regular uploads.

## Installation

```bash
# Install with XET support
uv pip install "huggingface_hub[xet]"

# Or with pip
pip install "huggingface_hub[xet]"
```

## Usage

### Quick Upload (after preparing dataset)

```bash
# Prepare dataset first
uv run tvpl-prepare-hf -r YOUR_USERNAME/vietnamese-legal-docs

# Upload with XET (automatic if available)
uv run python scripts/upload_with_xet.py -r YOUR_USERNAME/vietnamese-legal-docs
```

### Manual XET Upload

```python
from huggingface_hub.xet import upload_folder

upload_folder(
    repo_id="YOUR_USERNAME/vietnamese-legal-docs",
    repo_type="dataset",
    folder_path="data/hf_dataset",
)
```

### Regular Upload (fallback)

```bash
huggingface-cli upload data/hf_dataset YOUR_USERNAME/vietnamese-legal-docs --repo-type dataset
```

## Performance Comparison

| Method | Speed | Best For |
|--------|-------|----------|
| **XET** | ~100 MB/s | Large files (>500MB) |
| Regular | ~10-30 MB/s | Small to medium files |
| CLI | ~20 MB/s | Any size, easy to use |

## When to Use XET

✅ **Use XET when:**
- Dataset > 500MB
- Uploading many files
- Have reliable internet connection
- Speed is critical

❌ **Don't use XET when:**
- Dataset < 100MB (not worth overhead)
- Unstable connection
- First time uploading (test with regular method)

## Troubleshooting

**XET not available:**
```bash
pip install --upgrade "huggingface_hub[xet]"
```

**Upload timeout:**
```python
# Increase timeout
upload_folder(
    repo_id="...",
    folder_path="...",
    max_workers=4,  # Reduce workers
)
```

**Connection issues:**
- Switch to regular upload method
- Use HF CLI instead
- Try uploading smaller chunks
