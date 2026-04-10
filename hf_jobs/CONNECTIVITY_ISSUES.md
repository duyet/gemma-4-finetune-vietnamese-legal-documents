# HuggingFace Jobs Connectivity Issues

## Date
2026-04-10

## Problem
Persistent connectivity issues when trying to download models from HuggingFace/ModelScope on HF Jobs infrastructure.

## Jobs Affected
- 69d943cf - Tokenization stuck (padding='max_length' inefficiency)
- 69d947c9 - HF connection timeout (3x retries failed)
- 69d948aa - HF connection timeout (3x retries failed)
- 69d94988 - HF connection timeout (3x retries failed)
- 69d94c16 - No progress after 25+ minutes (ModelScope fallback)
- 69d950b6 - No progress after 10+ minutes (Phi-3-mini)
- 69d951f8 - AttributeError on checkpoint_files (local_files_only issue)
- 69d95274 - Still stuck at initial loading

## Root Causes Identified

1. **Network Connectivity**: HF Jobs servers cannot reach HuggingFace hub reliably
2. **ModelScope Fallback**: Using `UNSLOTH_USE_MODELSCOPE=1` doesn't resolve the issue
3. **Tokenization Bottleneck**: `padding='max_length'` creates 910M tokens (222K × 4096), taking 35+ minutes

## Fixes Attempted

1. ✅ **Dynamic padding** - Changed from `padding='max_length'` to `padding=False` + data collator
2. ✅ **Retry logic** - Added 3 retry attempts with exponential backoff
3. ✅ **ModelScope fallback** - Enabled `UNSLOTH_USE_MODELSCOPE=1`
4. ✅ **Model change** - Switched from Llama-3.2-3B to Phi-3-mini
5. ❌ **local_files_only** - Caused AttributeError on checkpoint_files

## Current Working Configuration
```python
# Tokenization (hf_jobs/train_unsloth_native.py)
tokenized = tokenizer(
    examples["text"],
    truncation=True,
    max_length=4096,
    padding=False,  # Collator will pad per batch
    return_tensors=None,
)

# Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
    padding="longest",  # Dynamic per-batch padding
)
```

## Alternative Approaches to Try

1. **Google Colab**: Better connectivity, can download models directly
2. **RunPod/AWS**: More reliable infrastructure
3. **Pre-cached models**: Use models already cached in HF Jobs Docker image
4. **Simpler models**: Try models with better availability
5. **Different framework**: Skip Unsloth, use plain transformers

## Jobs Affected (Continued)
- 69d95274 - Job RUNNING for 8+ hours with no progress after Triton warning

## Latest Analysis (2026-04-11)

**Job 69d95274** (Phi-3-mini-4k-instruct):
- Status: RUNNING since 2026-04-10 19:41:40 UTC
- Duration: 8+ hours with no training output
- Logs: Only shows "Unsloth: Will patch your computer" and Triton warning
- Issue: Script hangs during import/model loading, never reaches print statements
- Root cause: **Model loading timeout** - Unsloth's `FastLanguageModel.from_pretrained()` is stuck trying to download model from HuggingFace/ModelScope, but never completes and never times out explicitly

**Critical Finding**: The Triton error is just a warning. The real issue is that `from unsloth import FastLanguageModel` triggers model download which hangs indefinitely due to HF Jobs connectivity issues. The script never progresses to our print statements or retry logic because it's stuck at the import stage.

**Timeout Behavior**: Unsloth's model loading does NOT have a built-in timeout for network downloads. It will wait indefinitely for the model to download, which explains why jobs stay in "RUNNING" state for hours.

## Next Steps
1. **IMMEDIATE**: Switch to Google Colab - HF Jobs has fundamental connectivity issues
2. Continue autonomous monitoring on HF Jobs as backup (connectivity might improve)
3. Document successful Colab training run
4. Compare Colab vs HF Jobs performance and reliability
