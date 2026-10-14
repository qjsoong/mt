#!/usr/bin/env bash
# Download the released checkpoint and tokenizer from GitHub release assets.
#
# Usage: scripts/download_model.sh [tag] [destination]
#
# The repository is inferred from the "origin" remote; set MT_REPO to
# override (for example when working from a fork without release assets).
set -euo pipefail

TAG="${1:-v0.1.0}"
DEST="${2:-artifacts/release}"

if ! command -v gh >/dev/null 2>&1; then
    echo "error: the GitHub CLI (gh) is required" >&2
    exit 1
fi

REPO="${MT_REPO:-}"
if [ -z "${REPO}" ]; then
    remote_url="$(git remote get-url origin 2>/dev/null || true)"
    REPO="${remote_url#git@github.com:}"
    REPO="${REPO#https://github.com/}"
    REPO="${REPO%.git}"
fi
if [ -z "${REPO}" ]; then
    echo "error: could not determine the repository; set MT_REPO" >&2
    exit 1
fi

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
