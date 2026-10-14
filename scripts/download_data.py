#!/usr/bin/env python3
"""Download public parallel corpora and store them as JSONL.

The output files are line-delimited JSON objects::

    {"source": "...", "target": "...", "origin": "wmt14"}

``source`` is always German and ``target`` always English, matching the
orientation of the processed corpus. Corpora are *not* committed to the
repository; see ``docs/data.md`` for provenance and licensing notes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path


def iter_wmt14() -> Iterator[tuple[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset("wmt/wmt14", "de-en", split="train")
    for example in dataset:
        translation = example["translation"]
        yield translation["de"], translation["en"]


def iter_opus100() -> Iterator[tuple[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset("Helsinki-NLP/opus-100", "en-de", split="train")
    for example in dataset:
        translation = example["translation"]
        yield translation["de"], translation["en"]


SOURCES: dict[str, Iterable] = {
    "wmt14": iter_wmt14,
    "opus100": iter_opus100,
}


def download(names: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in SOURCES:
            raise SystemExit(
                f"unknown corpus {name!r}; available: {sorted(SOURCES)}"
            )
        output_path = output_dir / f"{name}.de-en.jsonl"
        if output_path.exists():
            print(f"skipping {output_path} (already exists)")
            continue
        print(f"downloading {name} -> {output_path}")
        count = 0
        with open(output_path, "w", encoding="utf-8") as handle:
            for source, target in SOURCES[name]():
                handle.write(
                    json.dumps(
                        {"source": source, "target": target, "origin": name},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
                if count % 500000 == 0:
                    print(f"  {count:,} pairs", flush=True)
        print(f"  {count:,} pairs written")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpora",
        default="wmt14,opus100",
        help="comma separated corpus names (wmt14, opus100)",
    )
    parser.add_argument("--output-dir", default="data/raw")
    arguments = parser.parse_args()
    download(
        [name.strip() for name in arguments.corpora.split(",") if name.strip()],
        Path(arguments.output_dir),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
