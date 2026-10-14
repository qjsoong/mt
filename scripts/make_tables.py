#!/usr/bin/env python3
"""Build markdown benchmark tables from evaluation JSON files.

Each input file has the shape produced by ``mt-eval`` and
``scripts/run_baseline.py``::

    {"model": "...", "parameters": 123, "results": [
        {"test_set": "wmt14", "direction": "de-en", "bleu": 1.0, ...}]}

Usage::

    python scripts/make_tables.py --inputs results/*.json --output results/summary.md
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def load_results(paths: list[Path]) -> list[dict]:
    payloads = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    return payloads


def format_parameters(parameters: int) -> str:
    return f"{parameters / 1e6:.1f}M" if parameters else "n/a"


def markdown_tables(payloads: list[dict]) -> str:
    directions = sorted(
        {
            result["direction"]
            for payload in payloads
            for result in payload["results"]
        }
    )
    test_sets = sorted(
        {
            result["test_set"]
            for payload in payloads
            for result in payload["results"]
        }
    )
    lines: list[str] = []
    for direction in directions:
        lines.append(f"### {direction.replace('-', '→')}")
        lines.append("")
        header = ["Model", "Params"]
        for test_set in test_sets:
            header.append(f"{test_set} BLEU")
            header.append(f"{test_set} chrF")
        header.append("sent/s")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + " --- |" * len(header))
        for payload in payloads:
            lookup = {
                (result["test_set"], result["direction"]): result
                for result in payload["results"]
            }
            row = [
                payload["model"],
                format_parameters(payload.get("parameters", 0)),
            ]
            speed = []
            for test_set in test_sets:
                result = lookup.get((test_set, direction))
                if result is None:
                    row.extend(["–", "–"])
                    continue
                row.append(f"{result['bleu']:.2f}")
                row.append(f"{result['chrf']:.2f}")
                if result.get("sentences_per_second"):
                    speed.append(result["sentences_per_second"])
            row.append(f"{sum(speed) / len(speed):.1f}" if speed else "–")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="evaluation JSON files (globs are expanded by the shell)",
    )
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()
    paths = [
        Path(path)
        for pattern in arguments.inputs
        for path in glob.glob(pattern)
    ]
    payloads = load_results(paths)
    table = markdown_tables(payloads)
    if arguments.output:
        output_path = Path(arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(table)
        print(f"wrote {output_path}")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
