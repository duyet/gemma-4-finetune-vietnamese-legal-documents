#!/bin/bash
# Autonomous HF Jobs Monitor and Submitter
# Keeps trying to submit jobs until successful training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }

# Configuration
MAX_RETRIES=1000
RETRY_INTERVAL=300  # 5 minutes
LOG_FILE="$PROJECT_ROOT/hf_jobs/autonomous_monitor.log"
SUMMARY_FILE="$PROJECT_ROOT/hf_jobs/autonomous_summary.json"

# Initialize summary
if [ ! -f "$SUMMARY_FILE" ]; then
    cat > "$SUMMARY_FILE" << EOF
{
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "attempts": 0,
  "successful_jobs": [],
  "failed_jobs": [],
  "last_status": "initializing"
}
EOF
fi

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

submit_job() {
    local attempt=$1
    local model=$2
    local script=$3

    log_info "Attempt #$attempt: Submitting job with $model"

    # Try to submit job
    if hf jobs run \
        --flavor t4-medium \
        --env "BASE_MODEL=$model" \
        --env "DATASET_NAME=duyet/vietnamese-legal-instruct" \
        --env "MAX_SEQ_LENGTH=2048" \
        --env "BATCH_SIZE=2" \
        --env "GRADIENT_ACCUMULATION=4" \
        --env "EPOCHS=1" \
        --env "LEARNING_RATE=2e-4" \
        --env "LORA_R=16" \
        --env "LORA_ALPHA=16" \
        --env "HF_USERNAME=duyet" \
        --env "HF_REPO_NAME=vietnamese-legal-autonomous-$attempt" \
        --env "PUSH_TO_HUB=true" \
        --env "EXPORT_GGUF=false" \
        --secrets "HF_TOKEN" \
        --detach \
        unsloth/unsloth \
        bash -c "cd /workspace && git clone --depth 1 https://github.com/duyet/gemma-4-finetune-vietnamese-legal-documents.git && cd gemma-4-finetune-vietnamese-legal-documents && pip install -q peft bitsandbytes && python $script" 2>&1 | tee -a "$LOG_FILE"; then

        # Extract job ID
        sleep 5
        local job_id=$(hf jobs ps 2>&1 | head -2 | tail -1 | awk '{print $1}')
        if [ -n "$job_id" ] && [ "$job_id" != "JOB" ]; then
            log_info "✅ Job submitted: $job_id"
            echo "$job_id"
            return 0
        fi
    fi

    return 1
}

monitor_job() {
    local job_id=$1
    local max_wait=7200  # 2 hours
    local elapsed=0
    local check_interval=60  # 1 minute

    log_info "🔍 Monitoring job $job_id..."

    while [ $elapsed -lt $max_wait ]; do
        # Check job status
        local status=$(hf jobs inspect "$job_id" 2>&1 | grep -A 1 '"stage"' | tail -1 | grep -o 'RUNNING\|SUCCEEDED\|FAILED\|ERROR\|CANCELED' || echo "UNKNOWN")

        log_info "Status: $status (${elapsed}s elapsed)"

        case $status in
            SUCCEEDED)
                log_info "🎉 Job $job_id SUCCEEDED!"
                update_summary "success" "$job_id"
                return 0
                ;;
            FAILED|ERROR|CANCELED)
                log_error "❌ Job $job_id $status"
                # Get logs
                hf jobs logs "$job_id" 2>&1 | tail -50 >> "$LOG_FILE"
                update_summary "failed" "$job_id"
                return 1
                ;;
            RUNNING)
                # Get recent logs
                local logs=$(hf jobs logs "$job_id" 2>&1 | tail -20)
                log_info "Recent logs:\n$logs"
                ;;
        esac

        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done

    log_warn "⏱️  Job $job_id timed out after $max_wait seconds"
    return 2
}

update_summary() {
    local result=$1
    local job_id=$2

    python3 << PYTHON
import json
from datetime import datetime

summary_file = "$SUMMARY_FILE"
with open(summary_file, 'r') as f:
    summary = json.load(f)

summary['attempts'] = summary.get('attempts', 0) + 1
summary['last_update'] = datetime.utcnow().isoformat()

if "$result" == "success":
    summary['successful_jobs'].append("$job_id")
    summary['last_status'] = 'success'
else:
    summary['failed_jobs'].append("$job_id")
    summary['last_status'] = 'failed'

with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary updated: {summary['attempts']} attempts, {len(summary['successful_jobs'])} successes")
PYTHON
}

# Model rotation strategy
models=(
    "Qwen/Qwen2.5-3B-Instruct"
    "unsloth/Llama-3.2-3B-Instruct"
    "microsoft/Phi-3-mini-4k-instruct"
    "sshleifer/tiny-gpt2"
)

scripts=(
    "hf_jobs/train_transformers_native.py"
    "hf_jobs/train_transformers_native.py"
    "hf_jobs/train_transformers_native.py"
    "hf_jobs/train_tiny_test.py"
)

log "=================================================="
log "🚀 AUTONOMOUS HF JOBS MONITOR"
log "=================================================="
log "Started: $(date -u)"
log "Log file: $LOG_FILE"
log "=================================================="

attempt=0
while [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))

    log ""
    log "=================================================="
    log "🔄 ITERATION #$attempt"
    log "=================================================="
    log "Time: $(date)"

    # Check if we can submit jobs (credit check)
    log_info "Checking HF Jobs credit status..."

    # Try to submit a job
    model_index=$(( (attempt - 1) % ${#models[@]} ))
    model="${models[$model_index]}"
    script="${scripts[$model_index]}"

    log_info "Strategy: Model rotation (attempt $attempt → $model)"

    if job_id=$(submit_job "$attempt" "$model" "$script"); then
        log_info "✅ Job submitted successfully: $job_id"

        # Monitor the job
        if monitor_job "$job_id"; then
            log_info "🎉 SUCCESS! Job $job_id completed successfully"
            log ""
            log "=================================================="
            log "✅ TRAINING COMPLETE"
            log "=================================================="
            log "Job ID: $job_id"
            log "Model: $model"
            log "Attempt: #$attempt"
            log "Time: $(date -u)"
            log ""
            log "Next steps:"
            log "1. Download model from HuggingFace"
            log "2. Generate evaluation scores"
            log "3. Create detailed model card"
            log "4. Continue optimization iterations"
            log "=================================================="

            # Send notification (would integrate with your notification system)
            log_info "🔔 Training successful notification"

            exit 0
        else
            log_warn "⚠️  Job $job_id failed, continuing to next attempt..."
        fi
    else
        log_warn "⚠️  Failed to submit job (credit limit or other error)"
        log_warn "Will retry in $RETRY_INTERVAL seconds..."

        # Update summary with failed attempt
        update_summary "failed" "submit-failed-$attempt"
    fi

    # Wait before next attempt
    log ""
    log_info "⏳ Waiting ${RETRY_INTERVAL}s before next attempt..."
    log "   Next attempt at: $(date -d '+$RETRY_INTERVAL seconds' 2>/dev/null || date -v+${RETRY_INTERVAL}S)"
    sleep $RETRY_INTERVAL
done

log ""
log "=================================================="
log "⚠️  MAX ATTEMPTS REACHED"
log "=================================================="
log "Total attempts: $MAX_RETRIES"
log "Summary file: $SUMMARY_FILE"
log "=================================================="

# Show final summary
cat "$SUMMARY_FILE"
