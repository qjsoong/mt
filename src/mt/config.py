"""Configuration objects for the mt model and training pipeline.

Configurations are plain dataclasses so that they can be serialized to and
from YAML/JSON without extra dependencies. Unknown keys raise an error on
purpose: a typo in a config file should fail loudly instead of silently
training with default hyperparameters.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class ModelConfig:
    """Architecture of the encoder--decoder Transformer."""

    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1408
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "swiglu"  # swiglu | gelu | relu
    norm: str = "rmsnorm"  # rmsnorm | layernorm
    positional: str = "sinusoidal"  # sinusoidal | learned
    max_positions: int = 1024
    tie_embeddings: bool = True
    padding_idx: int = 0
    bias: bool = False
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.activation not in {"swiglu", "gelu", "relu"}:
            raise ValueError(f"unknown activation: {self.activation}")
        if self.norm not in {"rmsnorm", "layernorm"}:
            raise ValueError(f"unknown norm: {self.norm}")
        if self.positional not in {"sinusoidal", "learned"}:
            raise ValueError(f"unknown positional: {self.positional}")


@dataclass
class DataConfig:
    """Where tokenized data lives and how batches are assembled."""

    data_dir: str = "data/processed"
    tokenizer_path: str = "artifacts/tokenizer/tokenizer.model"
    max_tokens: int = 16384
    max_batch_size: int = 256
    max_source_length: int = 128
    max_target_length: int = 128
    # "both" trains de->en and en->de alternately; otherwise a fixed direction.
    direction: str = "both"
    shuffle: bool = True
    sort_within_chunk: bool = True
    chunk_size: int = 10000

    def __post_init__(self) -> None:
        if self.direction not in {"both", "de-en", "en-de"}:
            raise ValueError(f"unknown direction: {self.direction}")


@dataclass
class TrainConfig:
    """Optimization and bookkeeping for a training run."""

    # Data
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Optimization
    max_steps: int = 40000
    max_epochs: int = 100
    lr: float = 0.0  # 0 means use the inverse-sqrt schedule peak lr
    lr_scale: float = 1.0
    schedule: str = "inverse_sqrt"  # inverse_sqrt | cosine
    warmup_steps: int = 4000
    min_lr_ratio: float = 0.05
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-9
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    grad_accum_steps: int = 1
    gradient_checkpointing: bool = False

    # Validation / checkpointing
    eval_every_steps: int = 2000
    eval_max_batches: int = 64
    log_every_steps: int = 50
    save_every_steps: int = 2000
    keep_step_checkpoints: int = 2
    patience: int = 0  # 0 disables early stopping

    # Misc
    out_dir: str = "artifacts/run"
    seed: int = 42
    device: str = "auto"
    amp_dtype: str = "bfloat16"  # bfloat16 | float16 | none
    decode_max_length: int = 200

    def __post_init__(self) -> None:
        if self.schedule not in {"inverse_sqrt", "cosine"}:
            raise ValueError(f"unknown schedule: {self.schedule}")
        if self.amp_dtype not in {"bfloat16", "float16", "none"}:
            raise ValueError(f"unknown amp dtype: {self.amp_dtype}")


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build a dataclass from a dict, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise TypeError(
            f"expected a mapping for {cls.__name__}, got {type(data)}"
        )
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            raise ValueError(
                f"unknown key {key!r} in {cls.__name__}; "
                f"valid keys: {sorted(known)}"
            )
        if key == "data":
            value = _from_dict(DataConfig, value)
        elif key == "model":
            value = _from_dict(ModelConfig, value)
        elif key == "betas" and isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> TrainConfig:
    """Load a :class:`TrainConfig` from a YAML file."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _from_dict(TrainConfig, raw)


def config_to_dict(config: Any) -> dict[str, Any]:
    """Recursively convert a config dataclass to a JSON/YAML friendly dict."""

    def convert(value: Any) -> Any:
        if dataclasses.is_dataclass(value):
            return {
                f.name: convert(getattr(value, f.name)) for f in fields(value)
            }
        if isinstance(value, tuple):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    return convert(config)


def save_config(config: Any, path: str | Path) -> None:
    """Write a config dataclass to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)
