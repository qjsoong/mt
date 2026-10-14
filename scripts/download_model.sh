#!/usr/bin/env bash
# Download the released checkpoint and tokenizer from GitHub release assets.
#
# Usage: scripts/download_model.sh [tag] [destination]
set -euo pipefail

TAG="${1:-v0.1.0}"
DEST="${2:-artifacts/release}"

if ! command -v gh >/dev/null 2>&1; then
    echo "error: the GitHub CLI (gh) is required" >&2
    exit 1
fi

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
mkdir -p "${DEST}"
echo "downloading release ${TAG} from ${REPO} -> ${DEST}"
gh release download "${TAG}" \
    --repo "${REPO}" \
    --dir "${DEST}" \
    --pattern "mt-small-deen.pt" \
    --pattern "tokenizer.model" \
    --pattern "model-config.yaml" \
    --clobber
ls -la "${DEST}"
