"""Small shared helpers: seeding, devices, logging."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def format_number(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}k"
    return f"{value:.0f}"


class JSONLLogger:
    """Append JSON-serializable dictionaries as lines to a log file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("time", time.time())
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


class ThroughputMeter:
    """Track tokens per second over a sliding window."""

    def __init__(self) -> None:
        self.start = time.time()
        self.tokens = 0

    def update(self, num_tokens: int) -> None:
        self.tokens += num_tokens

    def rate(self) -> float:
        elapsed = max(1e-6, time.time() - self.start)
        return self.tokens / elapsed

    def reset(self) -> None:
        self.start = time.time()
        self.tokens = 0


def count_files(directory: str | Path) -> int:
    return sum(1 for _ in Path(directory).iterdir())


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
