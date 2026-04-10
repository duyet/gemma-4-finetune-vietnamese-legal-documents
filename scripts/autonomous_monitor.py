#!/usr/bin/env python3
"""
Autonomous HF Jobs Monitor - Python version.
More reliable for background execution.
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
MAX_ATTEMPTS = 1000
RETRY_INTERVAL = 300  # 5 minutes
LOG_FILE = Path("hf_jobs/autonomous_monitor.log")
SUMMARY_FILE = Path("hf_jobs/autonomous_summary.json")

# Model rotation
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "sshleifer/tiny-gpt2",
]

SCRIPTS = [
    "hf_jobs/train_transformers_native.py",
    "hf_jobs/train_transformers_native.py",
    "hf_jobs/train_transformers_native.py",
    "hf_jobs/train_tiny_test.py",
]


def log(message, level="INFO"):
    """Log message to file and console."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{level}] {timestamp} - {message}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + '\n')


def submit_job(attempt, model, script):
    """Submit a HF Job."""
    log(f"Attempt #{attempt}: Submitting job with {model}")

    cmd = [
        "hf", "jobs", "run",
        "--flavor", "t4-medium",
        "--env", f"BASE_MODEL={model}",
        "--env", "DATASET_NAME=duyet/vietnamese-legal-instruct",
        "--env", "MAX_SEQ_LENGTH=2048",
        "--env", "BATCH_SIZE=2",
        "--env", "GRADIENT_ACCUMULATION=4",
        "--env", "EPOCHS=1",
        "--env", "LEARNING_RATE=2e-4",
        "--env", "LORA_R=16",
        "--env", "LORA_ALPHA=16",
        "--env", "HF_USERNAME=duyet",
        "--env", f"HF_REPO_NAME=vietnamese-legal-auto-{attempt}",
        "--env", "PUSH_TO_HUB=true",
        "--env", "EXPORT_GGUF=false",
        "--secrets", "HF_TOKEN",
        "--detach",
        "unsloth/unsloth",
        "bash", "-c",
        f"cd /workspace && git clone --depth 1 https://github.com/duyet/gemma-4-finetune-vietnamese-legal-documents.git && cd gemma-4-finetune-vietnamese-legal-documents && pip install -q peft bitsandbytes && python {script}"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Log output
        log(f"Submission output: {result.stdout}")
        if result.stderr:
            log(f"Submission errors: {result.stderr}")

        # Check for errors
        if "402 Payment Required" in result.stdout or "402 Payment Required" in result.stderr:
            log("❌ Credit limit - payment required", "ERROR")
            return None

        if result.returncode != 0:
            log(f"❌ Submission failed with code {result.returncode}", "ERROR")
            return None

        # Extract job ID
        for line in result.stdout.split('\n'):
            if "Job started with ID:" in line:
                job_id = line.split()[-1]
                log(f"✅ Job submitted: {job_id}")
                return job_id

        # Try to get from job list
        time.sleep(5)
        list_result = subprocess.run(
            ["hf", "jobs", "ps"],
            capture_output=True,
            text=True
        )

        for line in list_result.stdout.split('\n')[:3]:
            if line and not line.startswith('JOB'):
                job_id = line.split()[0]
                if len(job_id) == 24 and all(c in '0123456789abcdef' for c in job_id):
                    log(f"✅ Job found: {job_id}")
                    return job_id

        log("❌ Could not extract job ID", "ERROR")
        return None

    except subprocess.TimeoutExpired:
        log("❌ Submission timed out", "ERROR")
        return None
    except Exception as e:
        log(f"❌ Submission error: {e}", "ERROR")
        return None


def monitor_job(job_id, max_wait=7200):
    """Monitor a running job."""
    log(f"🔍 Monitoring job {job_id}...")

    elapsed = 0
    check_interval = 60

    while elapsed < max_wait:
        try:
            result = subprocess.run(
                ["hf", "jobs", "inspect", job_id],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Check for job not found
            if "Could not find job" in result.stdout or "Could not find job" in result.stderr:
                log(f"❌ Job {job_id} not found")
                return False

            # Get status
            for line in result.stdout.split('\n'):
                if '"stage"' in line:
                    if 'RUNNING' in result.stdout:
                        if elapsed % 300 == 0:  # Every 5 minutes
                            log(f"Status: RUNNING ({elapsed}s elapsed)")
                    elif 'SUCCEEDED' in result.stdout:
                        log(f"🎉 Job {job_id} SUCCEEDED!")
                        return True
                    elif any(x in result.stdout for x in ['FAILED"', 'ERROR"', 'CANCELED"']):
                        log(f"❌ Job {job_id} failed/canceled")
                        # Get logs
                        log_result = subprocess.run(
                            ["hf", "jobs", "logs", job_id],
                            capture_output=True,
                            text=True
                        )
                        log(f"Logs: {log_result.stdout[-500:]}")
                        return False

            elapsed += check_interval
            time.sleep(check_interval)

        except subprocess.TimeoutExpired:
            log(f"⚠️  Monitor check timed out")
            elapsed += check_interval
            time.sleep(check_interval)
        except Exception as e:
            log(f"⚠️  Monitor error: {e}")
            elapsed += check_interval
            time.sleep(check_interval)

    log(f"⏱️  Job {job_id} timed out after {max_wait}s")
    return False


def update_summary(status, job_id=None):
    """Update summary file."""
    if not SUMMARY_FILE.exists():
        summary = {
            "started_at": datetime.utcnow().isoformat(),
            "attempts": 0,
            "successful_jobs": [],
            "failed_jobs": [],
            "last_status": "initializing"
        }
    else:
        with open(SUMMARY_FILE) as f:
            summary = json.load(f)

    summary["attempts"] = summary.get("attempts", 0) + 1
    summary["last_update"] = datetime.utcnow().isoformat()

    if status == "success" and job_id:
        summary["successful_jobs"].append(job_id)
        summary["last_status"] = "success"
    else:
        if job_id:
            summary["failed_jobs"].append(job_id)
        else:
            summary["failed_jobs"].append(f"attempt-{summary['attempts']}")
        summary["last_status"] = "failed"

    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)

    log(f"Summary: {summary['attempts']} attempts, {len(summary['successful_jobs'])} successes")


def main():
    """Main autonomous loop."""
    log("="*60)
    log("🚀 AUTONOMOUS HF JOBS MONITOR (PYTHON)")
    log("="*60)
    log(f"Max attempts: {MAX_ATTEMPTS}")
    log(f"Retry interval: {RETRY_INTERVAL}s")
    log("="*60)

    attempt = 0

    while attempt < MAX_ATTEMPTS:
        attempt += 1

        log("")
        log("="*60)
        log(f"🔄 ITERATION #{attempt}")
        log("="*60)
        log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Model rotation
        model_index = (attempt - 1) % len(MODELS)
        model = MODELS[model_index]
        script = SCRIPTS[model_index]

        log(f"Strategy: Model rotation (attempt {attempt} → {model})")

        # Submit job
        job_id = submit_job(attempt, model, script)

        if job_id:
            log(f"✅ Job submitted successfully: {job_id}")

            # Monitor job
            if monitor_job(job_id):
                log("")
                log("="*60)
                log("🎉 SUCCESS! TRAINING COMPLETE")
                log("="*60)
                log(f"Job ID: {job_id}")
                log(f"Model: {model}")
                log(f"Attempt: #{attempt}")
                log("="*60)

                update_summary("success", job_id)

                # Run evaluation
                log("Running evaluation...")
                eval_result = subprocess.run(
                    [sys.executable, "scripts/evaluate_and_report.py"],
                    capture_output=True
                )
                if eval_result.returncode == 0:
                    log("✅ Evaluation complete")

                return 0
            else:
                log(f"⚠️  Job {job_id} failed, continuing...")
                update_summary("failed", job_id)
        else:
            log("⚠️  Failed to submit job (credit limit or other error)")
            log(f"Will retry in {RETRY_INTERVAL} seconds...")
            update_summary("failed", None)

        # Wait before next attempt
        log("")
        log(f"⏳ Waiting {RETRY_INTERVAL}s before next attempt...")
        log(f"Next attempt at: {datetime.fromtimestamp(time.time() + RETRY_INTERVAL).strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(RETRY_INTERVAL)

    log("")
    log("="*60)
    log("⚠️  MAX ATTEMPTS REACHED")
    log("="*60)

    # Show final summary
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE) as f:
            summary = json.load(f)
        log(f"Total attempts: {summary['attempts']}")
        log(f"Successful: {len(summary['successful_jobs'])}")
        log(f"Failed: {len(summary['failed_jobs'])}")

    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\n⚠️  Monitor interrupted by user")
        sys.exit(130)
    except Exception as e:
        log(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
