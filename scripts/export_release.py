#!/usr/bin/env python3
"""Export a training checkpoint as a compact inference checkpoint.

Training checkpoints contain optimizer state and RNG state, which roughly
triples their size. This script writes an inference-only checkpoint
containing just the weights, the model configuration and the tokenizer
path, and copies the tokenizer next to it.

Usage::

    python scripts/export_release.py \
        --checkpoint artifacts/run/best.pt \
        --output artifacts/release/mt-small-deen.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mt.config import config_to_dict  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer path; defaults to the path stored in the checkpoint",
    )
    arguments = parser.parse_args()

    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        arguments.checkpoint, map_location="cpu", weights_only=False
    )
    tokenizer_path = arguments.tokenizer or checkpoint.get("tokenizer_path")
    if tokenizer_path is None:
        raise SystemExit(
            "no tokenizer path in the checkpoint; pass --tokenizer"
        )

    slim = {
        "model": checkpoint["model"],
        "model_config": checkpoint["model_config"],
        "tokenizer_path": Path(tokenizer_path).name,
        "git_commit": checkpoint.get("git_commit"),
        "global_step": checkpoint.get("global_step"),
        "best_metric": checkpoint.get("best_metric"),
        "extra": checkpoint.get("extra", {}),
    }
    torch.save(slim, output_path)
    print(f"wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    config_path = output_path.with_name("model-config.yaml")
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config_to_dict(slim["model_config"]), handle, sort_keys=False
        )
    print(f"wrote {config_path}")

    tokenizer_output = output_path.with_name(Path(tokenizer_path).name)
    if Path(tokenizer_path).resolve() != tokenizer_output.resolve():
        shutil.copyfile(tokenizer_path, tokenizer_output)
    print(f"copied tokenizer to {tokenizer_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
