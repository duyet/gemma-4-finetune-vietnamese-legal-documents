#!/bin/bash
# Monitor HuggingFace Job with auto-refresh
# Usage: bash scripts/hf_jobs_monitor.sh <job-id>

JOB_ID="${1:-}"

if [ -z "$JOB_ID" ]; then
    echo "Usage: bash scripts/hf_jobs_monitor.sh <job-id>"
    echo ""
    echo "Getting latest job..."
    JOB_ID=$(hf jobs ps | grep -E "RUNNING|SCHEDULING" | head -1 | awk '{print $1}')
    if [ -z "$JOB_ID" ]; then
        echo "No active jobs found"
        exit 1
    fi
    echo "Using job: $JOB_ID"
    echo ""
fi

echo "🔍 Monitoring HF Job: $JOB_ID"
echo "================================"
echo ""

# Get job URL
JOB_URL=$(hf jobs inspect "$JOB_ID" 2>&1 | grep -o '"url": "[^"]*"' | cut -d'"' -f4)
echo "Job URL: $JOB_URL"
echo ""

while true; do
    clear
    echo "🔍 Monitoring HF Job: $JOB_ID"
    echo "================================"
    echo "Last update: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Get job status
    STATUS=$(hf jobs inspect "$JOB_ID" 2>&1)

    # Parse status
    STAGE=$(echo "$STATUS" | grep -o '"stage": "[^"]*"' | cut -d'"' -f4)
    MESSAGE=$(echo "$STATUS" | grep -o '"message": "[^"]*"' | cut -d'"' -f4 | sed 's/\\n/ /g')

    echo "Status: $STAGE"
    if [ "$MESSAGE" != "null" ] && [ -n "$MESSAGE" ]; then
        echo "Message: $MESSAGE"
    fi
    echo ""

    # Show recent logs if running
    if [ "$STAGE" = "RUNNING" ]; then
        echo "📋 Recent logs (last 20 lines):"
        echo "--------------------------------"
        hf jobs logs "$JOB_ID" 2>&1 | tail -20
    fi

    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "Or wait 30s for next check..."

    # Exit if job completed or failed
    if [ "$STAGE" = "FINISHED" ] || [ "$STAGE" = "FAILED" ] || [ "$STAGE" = "ERROR" ]; then
        echo ""
        echo "🏁 Job $STAGE!"
        echo "View details: $JOB_URL"
        break
    fi

    sleep 30
done
