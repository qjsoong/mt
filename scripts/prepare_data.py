#!/usr/bin/env python3
"""Turn raw JSONL corpora into tokenized, memory-mapped training data.

Example::

    python scripts/prepare_data.py \
        --raw-dir data/raw \
        --out-dir data/processed \
        --tokenizer artifacts/tokenizer/tokenizer.model
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mt.data import PrepareConfig, prepare_corpus  # noqa: E402


def iter_raw_pairs(raw_dir: Path) -> Iterator[tuple[str, str, str]]:
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no .jsonl files found in {raw_dir}")
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                yield record["source"], record["target"], record["origin"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument(
        "--tokenizer", default="artifacts/tokenizer/tokenizer.model"
    )
    parser.add_argument("--name", default="wmt14+opus100 de-en")
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--max-characters", type=int, default=512)
    parser.add_argument("--min-length-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--max-length-ratio", type=float, default=3.0)
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=0.0007,
        help="uniform hash-based fraction of pairs held out for validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = no limit")
    arguments = parser.parse_args()

    config = PrepareConfig(
        out_dir=arguments.out_dir,
        tokenizer_path=arguments.tokenizer,
        source_name=arguments.name,
        max_source_length=arguments.max_source_length,
        max_target_length=arguments.max_target_length,
        max_characters=arguments.max_characters,
        min_length_ratio=arguments.min_length_ratio,
        max_length_ratio=arguments.max_length_ratio,
        dev_fraction=arguments.dev_fraction,
        seed=arguments.seed,
        num_workers=arguments.num_workers,
        max_pairs=arguments.max_pairs,
    )
    print(f"preparing corpus -> {config.out_dir}")
    meta = prepare_corpus(iter_raw_pairs(Path(arguments.raw_dir)), config)
    print(
        "done: "
        f"{meta['num_train_pairs']:,} train pairs, "
        f"{meta['num_valid_pairs']:,} valid pairs, "
        f"{meta['num_train_src_tokens']:,} source tokens, "
        f"{meta['num_train_tgt_tokens']:,} target tokens"
    )
    print(f"rejected: {meta['rejected']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
