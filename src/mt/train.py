"""Training entry point.

Usage::

    mt-train --config configs/mt-small.yaml

The script is resumable: ``mt-train --config ... --resume artifacts/run/last.pt``
picks up the optimizer state, step counters and RNG state.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .config import (
    TrainConfig,
    config_to_dict,
    load_config,
    save_config,
)
from .data import Batch, TokenizedCorpus, iter_batches, load_meta
from .model import Transformer
from .tokenizer import MTTokenizer
from .utils import (
    JSONLLogger,
    ThroughputMeter,
    format_number,
    get_git_commit,
    resolve_device,
    set_seed,
)

AMP_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "none": None,
}


# ----------------------------------------------------------------------
# Optimization helpers
# ----------------------------------------------------------------------
def inverse_sqrt_lr(step: int, d_model: int, warmup: int) -> float:
    """The schedule from *Attention Is All You Need*."""
    step = max(1, step)
    return (d_model**-0.5) * min(step * warmup**-1.5, step**-0.5)


def cosine_lr(
    step: int, max_steps: int, warmup: int, min_ratio: float
) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def learning_rate(step: int, config: TrainConfig) -> float:
    d_model = config.model.d_model
    if config.lr > 0:
        peak = config.lr
        if config.schedule == "inverse_sqrt":
            factor = (
                min(
                    max(1, step) * config.warmup_steps**-1.5,
                    max(1, step) ** -0.5,
                )
                * (d_model**-0.5) ** -1
            )
        else:
            factor = cosine_lr(
                step, config.max_steps, config.warmup_steps, config.min_lr_ratio
            )
        return peak * factor * config.lr_scale
    return inverse_sqrt_lr(step, d_model, config.warmup_steps) * config.lr_scale


def make_optimizer(model: Transformer, config: TrainConfig) -> AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() <= 1 or "embedding" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(
        groups,
        lr=1.0,
        betas=config.betas,
        eps=config.eps,
    )


def compute_loss(
    model: Transformer,
    batch: Batch,
    label_smoothing: float,
    pad_id: int,
) -> tuple[torch.Tensor, int]:
    logits = model(
        batch.source,
        batch.target_input,
        batch.source_padding_mask,
        batch.target_padding_mask,
    )
    vocabulary = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, vocabulary),
        batch.target_output.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )
    num_tokens = int((batch.target_output != pad_id).sum())
    return loss, num_tokens


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_loss(
    model: Transformer,
    corpus: TokenizedCorpus,
    tokenizer_meta: dict,
    config: TrainConfig,
    device: torch.device,
    max_batches: int = 64,
) -> float:
    model.eval()
    amp_dtype = AMP_DTYPES[config.amp_dtype]
    total_loss = 0.0
    total_tokens = 0
    batches = iter_batches(
        corpus,
        tokenizer_meta,
        direction=config.data.direction,
        max_tokens=config.data.max_tokens,
        max_batch_size=config.data.max_batch_size,
        shuffle=False,
        sort_within_chunk=True,
        chunk_size=10000,
        seed=config.seed,
        epoch=0,
    )
    for index, batch in enumerate(batches):
        if index >= max_batches:
            break
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None and device.type == "cuda",
        ):
            loss, num_tokens = compute_loss(
                model,
                batch,
                config.label_smoothing,
                tokenizer_meta["tokenizer"]["pad_id"],
            )
        total_loss += float(loss) * num_tokens
        total_tokens += num_tokens
    model.train()
    return total_loss / max(1, total_tokens)


def move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        source=batch.source.to(device, non_blocking=True),
        target_input=batch.target_input.to(device, non_blocking=True),
        target_output=batch.target_output.to(device, non_blocking=True),
        source_padding_mask=batch.source_padding_mask.to(
            device, non_blocking=True
        ),
        target_padding_mask=batch.target_padding_mask.to(
            device, non_blocking=True
        ),
        direction=batch.direction,
        num_target_tokens=batch.num_target_tokens,
    )


def save_checkpoint(
    path: Path,
    model: Transformer,
    optimizer: AdamW,
    config: TrainConfig,
    tokenizer_path: str,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    best_metric: float,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": config_to_dict(config.model),
        "train_config": config_to_dict(config),
        "tokenizer_path": tokenizer_path,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "git_commit": get_git_commit(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "extra": extra or {},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


# ----------------------------------------------------------------------
# Main training routine
# ----------------------------------------------------------------------
def train(
    config: TrainConfig, tokenizer_path: str, resume: str | None = None
) -> Path:
    # Reduces allocator fragmentation on GPUs with little headroom.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")
    logger = JSONLLogger(out_dir / "log.jsonl")

    data_meta = load_meta(config.data.data_dir)
    tokenizer = MTTokenizer(tokenizer_path)
    if tokenizer.vocab_size != config.model.vocab_size:
        raise ValueError(
            f"tokenizer vocab size {tokenizer.vocab_size} does not match "
            f"configured model vocab size {config.model.vocab_size}; "
            "update config.model.vocab_size"
        )
    train_corpus = TokenizedCorpus(config.data.data_dir, "train")
    valid_corpus = TokenizedCorpus(config.data.data_dir, "valid")

    model = Transformer(config.model).to(device)
    model.set_gradient_checkpointing(config.gradient_checkpointing)
    optimizer = make_optimizer(model, config)
    amp_dtype = AMP_DTYPES[config.amp_dtype]
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )

    start_epoch = 0
    start_batch = 0
    global_step = 0
    best_metric = float("inf")
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint["batch_in_epoch"]
        global_step = checkpoint["global_step"]
        best_metric = checkpoint.get("best_metric", float("inf"))
        if checkpoint.get("torch_rng") is not None:
            torch.set_rng_state(checkpoint["torch_rng"])
        if checkpoint.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        print(f"resumed from {resume} at step {global_step}")

    print(
        f"model: {model.num_parameters():,} parameters | "
        f"train pairs: {len(train_corpus):,} | "
        f"valid pairs: {len(valid_corpus):,} | device: {device}"
    )
    logger.log(
        {
            "event": "start",
            "parameters": model.num_parameters(),
            "train_pairs": len(train_corpus),
            "valid_pairs": len(valid_corpus),
            "device": str(device),
            "git_commit": get_git_commit(),
            "data_meta": {
                key: data_meta[key]
                for key in ("name", "num_train_pairs", "num_train_src_tokens")
                if key in data_meta
            },
        }
    )

    pad_id = tokenizer.pad_id
    model.train()
    meter = ThroughputMeter()
    running_loss = 0.0
    start_time = time.time()
    keep_checkpoints: list[Path] = []
    epoch = start_epoch
    batch_index = 0
    last_loss: float | None = None
    evals_without_improvement = 0

    def flush_optimizer() -> float:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip
        )
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(gradient_norm)

    def should_stop() -> bool:
        return global_step >= config.max_steps

    for epoch in range(start_epoch, config.max_epochs):
        if should_stop():
            break
        pending_gradients = 0
        batches = iter_batches(
            train_corpus,
            data_meta,
            direction=config.data.direction,
            max_tokens=config.data.max_tokens,
            max_batch_size=config.data.max_batch_size,
            shuffle=config.data.shuffle,
            sort_within_chunk=config.data.sort_within_chunk,
            chunk_size=config.data.chunk_size,
            seed=config.seed,
            epoch=epoch,
        )
        for batch_index, batch in enumerate(batches):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            if should_stop():
                break
            batch = move_batch(batch, device)
            lr = learning_rate(global_step + 1, config)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=use_amp,
            ):
                loss, num_tokens = compute_loss(
                    model, batch, config.label_smoothing, pad_id
                )
                scaled_loss = loss / config.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            pending_gradients += 1

            is_update = pending_gradients >= config.grad_accum_steps
            if is_update:
                gradient_norm = flush_optimizer()
                pending_gradients = 0
                global_step += 1

                running_loss += float(loss.detach())
                last_loss = float(loss.detach())
                meter.update(num_tokens)

                if global_step % config.log_every_steps == 0:
                    average_loss = running_loss / config.log_every_steps
                    record = {
                        "event": "train",
                        "step": global_step,
                        "epoch": epoch,
                        "loss": float(loss.detach()),
                        "avg_loss": average_loss,
                        "lr": lr,
                        "grad_norm": gradient_norm,
                        "tokens_per_second": meter.rate(),
                        "elapsed": time.time() - start_time,
                    }
                    logger.log(record)
                    print(
                        f"step {global_step:>7} | epoch {epoch:>3} | "
                        f"loss {float(loss.detach()):.4f} | lr {lr:.2e} | "
                        f"grad {gradient_norm:.2f} | "
                        f"{format_number(meter.rate())} tok/s"
                    )
                    running_loss = 0.0
                    meter.reset()

            if (
                config.eval_every_steps
                and global_step % config.eval_every_steps == 0
            ):
                validation_loss = evaluate_loss(
                    model,
                    valid_corpus,
                    data_meta,
                    config,
                    device,
                    config.eval_max_batches,
                )
                record = {
                    "event": "eval",
                    "step": global_step,
                    "epoch": epoch,
                    "valid_loss": validation_loss,
                }
                print(
                    f"step {global_step:>7} | valid loss {validation_loss:.4f}"
                )
                logger.log(record)
                metric = validation_loss
                if metric < best_metric:
                    best_metric = metric
                    evals_without_improvement = 0
                    save_checkpoint(
                        out_dir / "best.pt",
                        model,
                        optimizer,
                        config,
                        tokenizer_path,
                        epoch,
                        batch_index + 1,
                        global_step,
                        best_metric,
                        extra={"valid_loss": validation_loss},
                    )
                    print(
                        f"  new best checkpoint (valid loss {validation_loss:.4f})"
                    )
                else:
                    evals_without_improvement += 1
                    if (
                        config.patience
                        and evals_without_improvement >= config.patience
                    ):
                        print(
                            f"early stopping after {config.patience} evals "
                            "without improvement"
                        )
                        break

            if (
                config.save_every_steps
                and global_step % config.save_every_steps == 0
            ):
                checkpoint_path = out_dir / f"step_{global_step:07d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    config,
                    tokenizer_path,
                    epoch,
                    batch_index + 1,
                    global_step,
                    best_metric,
                )
                keep_checkpoints.append(checkpoint_path)
                while len(keep_checkpoints) > config.keep_step_checkpoints:
                    stale = keep_checkpoints.pop(0)
                    stale.unlink(missing_ok=True)

        if pending_gradients:
            # Flush a partial accumulation window at the end of the epoch.
            gradient_norm = flush_optimizer()
            pending_gradients = 0
            global_step += 1

    save_checkpoint(
        out_dir / "last.pt",
        model,
        optimizer,
        config,
        tokenizer_path,
        epoch,
        batch_index + 1,
        global_step,
        best_metric,
    )
    if not (out_dir / "best.pt").exists():
        save_checkpoint(
            out_dir / "best.pt",
            model,
            optimizer,
            config,
            tokenizer_path,
            epoch,
            batch_index + 1,
            global_step,
            best_metric,
        )
    summary = {
        "event": "done",
        "steps": global_step,
        "best_metric": best_metric,
        "elapsed": time.time() - start_time,
        "final_loss": last_loss,
    }
    logger.log(summary)
    print(
        f"training finished: {global_step} steps, best valid loss "
        f"{best_metric:.4f}, elapsed {summary['elapsed'] / 3600:.2f} h"
    )
    return out_dir / "best.pt"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the mt translation model"
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--resume", default=None, help="checkpoint to resume from"
    )
    parser.add_argument(
        "--out-dir", default=None, help="override output directory"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="override max steps"
    )
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    if arguments.out_dir:
        config.out_dir = arguments.out_dir
    if arguments.max_steps is not None:
        config.max_steps = arguments.max_steps
    train(config, config.data.tokenizer_path, resume=arguments.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
