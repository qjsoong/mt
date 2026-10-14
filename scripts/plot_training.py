#!/usr/bin/env python3
"""Plot training curves from a run's ``log.jsonl``.

Usage::

    python scripts/plot_training.py \
        --log artifacts/run/log.jsonl \
        --output results/training-curve.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_log(path: Path) -> dict[str, list[float]]:
    steps, losses, validation_steps, validation_losses = [], [], [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "train":
                steps.append(record["step"])
                losses.append(record["avg_loss"])
            elif record.get("event") == "eval":
                validation_steps.append(record["step"])
                validation_losses.append(record["valid_loss"])
    return {
        "steps": steps,
        "losses": losses,
        "validation_steps": validation_steps,
        "validation_losses": validation_losses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="artifacts/run/log.jsonl")
    parser.add_argument("--output", default="results/training-curve.png")
    arguments = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        return 1

    data = read_log(Path(arguments.log))
    if not data["steps"]:
        print(f"no training records in {arguments.log}", file=sys.stderr)
        return 1

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(data["steps"], data["losses"], linewidth=0.6, alpha=0.6)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("training loss (batch avg)")
    axes[0].set_title("Training loss")
    if data["validation_steps"]:
        axes[1].plot(
            data["validation_steps"],
            data["validation_losses"],
            marker="o",
            markersize=2,
        )
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("validation loss (per token)")
    axes[1].set_title("Validation loss")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()

    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
