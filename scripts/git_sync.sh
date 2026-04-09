#!/bin/bash
# Git Automation Script for Gemma 4 Vietnamese Legal Documents
# Commit and push changes with semantic commits
# Supports dual-sync: GitHub + HuggingFace

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

# Load .env file if it exists
load_env() {
    local env_file="$PROJECT_ROOT/.env"
    if [ -f "$env_file" ]; then
        log_info "Loading environment from $env_file"
        set -a
        source "$env_file"
        set +a
    else
        log_warn ".env file not found. Using defaults or environment variables."
    fi
}

# Load environment variables
load_env

# Repository configuration (can be overridden by .env)
GITHUB_REPO="${GITHUB_REPO:-git@github.com:duyet/gemma-4-finetune-vietnamese-legal-documents.git}"
HF_REPO="${HF_USERNAME:-duyet}/${HF_REPO_NAME:-gemma-4-finetune-vietnamese-legal-documents}"

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

# Setup remotes (GitHub + HuggingFace)
setup_remotes() {
    log_info "Setting up dual remotes (GitHub + HuggingFace)..."

    # GitHub remote (origin)
    if git remote get-url origin &>/dev/null; then
        log_info "✅ GitHub remote (origin) already configured"
    else
        log_info "Adding GitHub remote..."
        git remote add origin "$GITHUB_REPO"
        log_info "✅ Added GitHub remote"
    fi

    # HuggingFace remote (hf)
    if git remote get-url hf &>/dev/null; then
        log_info "✅ HuggingFace remote (hf) already configured"
    else
        log_info "Adding HuggingFace remote..."
        git remote add hf "https://huggingface.co/$HF_REPO"
        log_info "✅ Added HuggingFace remote"
    fi

    log_info "Current remotes:"
    git remote -v
}

# Push to GitHub
push_github() {
    local branch="${1:-$(git rev-parse --abbrev-ref HEAD)}"

    if ! git rev-parse --verify "origin/$branch" > /dev/null 2>&1; then
        log_warn "GitHub branch 'origin/$branch' does not exist"
        log_info "Setting upstream and pushing to GitHub..."
        git push -u origin "$branch"
    else
        log_cmd "git push origin $branch"
        git push origin "$branch"
    fi

    log_info "✅ Pushed to GitHub (origin/$branch)"
}

# Push to HuggingFace
push_huggingface() {
    local branch="${1:-$(git rev-parse --abbrev-ref HEAD)}"

    # Check if HF remote exists
    if ! git remote get-url hf &>/dev/null; then
        log_warn "HuggingFace remote not configured. Run: $0 setup"
        return 1
    fi

    if ! git rev-parse --verify "hf/$branch" > /dev/null 2>&1; then
        log_warn "HuggingFace branch 'hf/$branch' does not exist"
        log_info "Setting upstream and pushing to HuggingFace..."
        git push -u hf "$branch"
    else
        log_cmd "git push hf $branch"
        git push hf "$branch"
    fi

    log_info "✅ Pushed to HuggingFace (hf/$branch)"
}

# Push to both GitHub and HuggingFace
push_dual() {
    local branch="${1:-$(git rev-parse --abbrev-ref HEAD)}"

    log_info "🔄 Pushing to both GitHub and HuggingFace..."

    push_github "$branch"
    push_huggingface "$branch"

    log_info "✅ Dual-sync complete!"
}

# Push to remote (legacy compatibility)
push_changes() {
    push_dual "$@"
}

# Show usage
usage() {
    cat << EOF
Git Automation for Gemma 4 Vietnamese Legal Documents

Usage: $(basename "$0") [COMMAND] [OPTIONS]

Commands:
    setup                    Setup dual remotes (GitHub + HF)
    commit                   Commit all changes
    commit [scope] msg       Commit with semantic message
    push                     Push to both GitHub and HF
    push github              Push to GitHub only
    push hf                  Push to HuggingFace only
    sync                     Commit and push to both
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
    # Setup remotes first
    $(basename "$0") setup

    # Quick commit and push to both
    $(basename "$0") sync

    # Commit with scope
    $(basename "$0") commit crawler "Add parallel crawling support"

    # Push to GitHub only
    $(basename "$0") push github

    # Push to HuggingFace only
    $(basename "$0") push hf

    # Show status
    $(basename "$0") status

Environment Variables:
    HF_USERNAME              Your HuggingFace username (default: duyet)
EOF
}

# Main
main() {
    cd "$PROJECT_ROOT"
    check_git

    local command="$1"
    shift || true

    case "$command" in
        setup)
            setup_remotes
            ;;

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
            local target="${1:-both}"
            shift  # Remove target from arguments so remaining args can be passed to branch parameter
            case "$target" in
                github)
                    push_github "$@"
                    ;;
                hf|huggingface)
                    push_huggingface "$@"
                    ;;
                both|"")
                    push_dual "$@"
                    ;;
                *)
                    log_error "Unknown push target: $target"
                    echo "Use: github, hf, or both"
                    exit 1
                    ;;
            esac
            ;;

        sync)
            stage_files
            commit_changes "chore" "Auto-sync"
            push_dual
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
            echo "  1) Setup remotes (GitHub + HF)"
            echo "  2) Commit (with scope and message)"
            echo "  3) Quick commit (default: chore)"
            echo "  4) Push to both"
            echo "  5) Push to GitHub only"
            echo "  6) Push to HuggingFace only"
            echo "  7) Sync (commit + push to both)"
            echo "  8) Status"
            echo ""
            read -p "Choice [1-8]: " choice

            case $choice in
                1)
                    setup_remotes
                    ;;
                2)
                    read -p "Scope: " scope
                    read -p "Message: " msg
                    stage_files
                    commit_changes "$scope" "$msg"
                    ;;
                3)
                    read -p "Message: " msg
                    stage_files
                    commit_changes "chore" "$msg"
                    ;;
                4)
                    push_dual
                    ;;
                5)
                    push_github
                    ;;
                6)
                    push_huggingface
                    ;;
                7)
                    stage_files
                    commit_changes "chore" "Auto-sync"
                    push_dual
                    ;;
                8)
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
