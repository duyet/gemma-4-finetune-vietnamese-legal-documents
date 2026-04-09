#!/bin/bash
# TVPL Git Automation Script
# Commit and push changes with semantic commits

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_cmd() { echo -e "${BLUE}[CMD]${NC} $1"; }

# Check if in git repo
check_git() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not a git repository"
        exit 1
    fi
}

# Get git status
get_status() {
    git status --porcelain
}

# Stage files
stage_files() {
    local files=("$@")

    if [ ${#files[@]} -eq 0 ]; then
        # Stage all changes
        log_cmd "git add -A"
        git add -A
    else
        # Stage specific files
        for file in "${files[@]}"; do
            log_cmd "git add $file"
            git add "$file"
        done
    fi
}

# Create semantic commit
commit_changes() {
    local scope="$1"
    local message="$2"

    # Check if there are changes
    if [ -z "$(get_status)" ]; then
        log_warn "No changes to commit"
        return 0
    fi

    # Format commit message
    if [ -n "$scope" ]; then
        commit_msg="$scope: $message"
    else
        commit_msg="$message"
    fi

    # Commit
    log_cmd "git commit -m \"$commit_msg\""
    git commit -m "$commit_msg"

    log_info "✅ Committed: $commit_msg"
}

# Push to remote
push_changes() {
    local branch="${1:-$(git rev-parse --abbrev-ref HEAD)}"

    # Check if remote exists
    if ! git rev-parse --verify "origin/$branch" > /dev/null 2>&1; then
        log_warn "Remote branch 'origin/$branch' does not exist"
        log_info "Setting upstream and pushing..."
        git push -u origin "$branch"
    else
        log_cmd "git push origin $branch"
        git push origin "$branch"
    fi

    log_info "✅ Pushed to origin/$branch"
}

# Show usage
usage() {
    cat << EOF
TVPL Git Automation

Usage: $(basename "$0") [COMMAND] [OPTIONS]

Commands:
    commit                   Commit all changes
    commit [scope] msg       Commit with semantic message
    push                     Push to remote
    sync                     Commit and push
    status                   Show git status
    diff                     Show changes

Commit Scopes:
    crawler                  Crawler changes
    data                     Data processing
    rag                      RAG pipeline
    docs                     Documentation
    chore                    Maintenance
    feat                     New feature
    fix                      Bug fix

Examples:
    # Quick commit and push
    $(basename "$0") sync

    # Commit with scope
    $(basename "$0") commit crawler "Add parallel crawling support"

    # Push only
    $(basename "$0") push

    # Show status
    $(basename "$0") status
EOF
}

# Main
main() {
    cd "$PROJECT_ROOT"
    check_git

    local command="$1"
    shift || true

    case "$command" in
        commit)
            local scope="$1"
            local message="$2"
            if [ -z "$message" ]; then
                # Interactive commit
                log_cmd "git commit"
                git commit
            else
                stage_files
                commit_changes "$scope" "$message"
            fi
            ;;

        push)
            push_changes "$@"
            ;;

        sync)
            stage_files
            commit_changes "chore" "Update project files"
            push_changes
            ;;

        status)
            git status
            ;;

        diff)
            git diff
            ;;

        "")
            # Interactive mode
            echo "Choose an action:"
            echo "  1) Commit (with scope and message)"
            echo "  2) Quick commit (default: chore)"
            echo "  3) Push"
            echo "  4) Sync (commit + push)"
            echo "  5) Status"
            echo ""
            read -p "Choice [1-5]: " choice

            case $choice in
                1)
                    read -p "Scope: " scope
                    read -p "Message: " msg
                    stage_files
                    commit_changes "$scope" "$msg"
                    ;;
                2)
                    read -p "Message: " msg
                    stage_files
                    commit_changes "chore" "$msg"
                    ;;
                3)
                    push_changes
                    ;;
                4)
                    stage_files
                    commit_changes "chore" "Auto-sync"
                    push_changes
                    ;;
                5)
                    git status
                    ;;
                *)
                    log_error "Invalid choice"
                    exit 1
                    ;;
            esac
            ;;

        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"
