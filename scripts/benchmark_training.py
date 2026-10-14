#!/usr/bin/env python3
"""Measure training throughput for different batch sizes.

Example::

    python scripts/benchmark_training.py \
        --config configs/mt-small-deen.yaml \
        --data-dir data/processed \
        --max-tokens 8192 16384 32768

Prints target tokens/second and peak GPU memory for each batch size. The
first steps are discarded as warm-up.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mt.config import load_config  # noqa: E402
from mt.data import TokenizedCorpus, iter_batches, load_meta  # noqa: E402
from mt.model import Transformer  # noqa: E402
from mt.train import compute_loss, make_optimizer, move_batch  # noqa: E402
from mt.utils import resolve_device, set_seed  # noqa: E402


def benchmark(
    config,
    data_dir: str,
    max_tokens: int,
    steps: int,
    warmup: int,
    gradient_checkpointing: bool = False,
) -> dict:
    device = resolve_device(config.device)
    if device.type != "cuda":
        raise SystemExit("training benchmark requires a CUDA device")
    set_seed(config.seed)
    meta = load_meta(data_dir)
    corpus = TokenizedCorpus(data_dir, "train")
    model = Transformer(config.model).to(device)
    model.set_gradient_checkpointing(gradient_checkpointing)
    optimizer = make_optimizer(model, config)
    amp_dtype = (
        torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
    )
    model.train()
    torch.cuda.reset_peak_memory_stats(device)

    batches = iter_batches(
        corpus,
        meta,
        direction=config.data.direction,
        max_tokens=max_tokens,
        max_batch_size=config.data.max_batch_size,
        shuffle=True,
        sort_within_chunk=config.data.sort_within_chunk,
        chunk_size=config.data.chunk_size,
        seed=config.seed,
        epoch=0,
    )
    total_tokens = 0
    start = time.time()
    for step, batch in enumerate(batches):
        if step >= steps:
            break
        batch = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            loss, num_tokens = compute_loss(
                model,
                batch,
                config.label_smoothing,
                meta["tokenizer"]["pad_id"],
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step >= warmup:
            total_tokens += num_tokens
    torch.cuda.synchronize(device)
    elapsed = time.time() - start
    return {
        "max_tokens": max_tokens,
        "steps": steps,
        "tokens_per_second": total_tokens / max(1e-9, elapsed),
        "seconds_per_step": elapsed / max(1, steps),
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1e6,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mt-small-deen.yaml")
    parser.add_argument("--data-dir", default=None, help="defaults to config")
    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        default=[8192, 16384, 32768],
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="enable activation checkpointing during the benchmark",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    data_dir = arguments.data_dir or config.data.data_dir
    print(f"model parameters: {Transformer(config.model).num_parameters():,}")
    print(f"device: {resolve_device(config.device)}")
    for max_tokens in arguments.max_tokens:
        try:
            result = benchmark(
                config,
                data_dir,
                max_tokens,
                arguments.steps,
                arguments.warmup,
                gradient_checkpointing=arguments.gradient_checkpointing,
            )
        except torch.OutOfMemoryError:
            print(f"max_tokens={max_tokens:>6} | CUDA out of memory")
            torch.cuda.empty_cache()
            continue
        print(
            f"max_tokens={result['max_tokens']:>6} | "
            f"{result['tokens_per_second']:>8.0f} target tok/s | "
            f"{result['seconds_per_step']:.3f} s/step | "
            f"peak {result['peak_memory_mb']:.0f} MB"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
