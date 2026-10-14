"""Data preparation and batching.

The processed format stores all tokenized sentences of one corpus split in
flat memory-mapped arrays::

    <split>.src.bin        uint16 token ids, all source sentences
    <split>.src.offsets    int64 start offsets, length N + 1
    <split>.tgt.bin        uint16 token ids, all target sentences
    <split>.tgt.offsets    int64 start offsets, length N + 1

The raw corpus is stored in the WMT orientation: ``src`` is German and
``tgt`` is English. Training on ``en-de`` simply swaps the two sides and
prepends the appropriate target-language token, so both directions share the
same storage.
"""

from __future__ import annotations

import html
import json
import multiprocessing as mp
import re
import unicodedata
import zlib
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .tokenizer import MTTokenizer

WHITESPACE = re.compile(r"\s+")
CONTROL_CHARS = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f-\u009f]")
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)
DEV_MODULUS = 10000


# ----------------------------------------------------------------------
# Text normalization and filtering
# ----------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Unicode-normalize, unescape HTML entities and collapse whitespace."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = text.translate(ZERO_WIDTH)
    text = CONTROL_CHARS.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def is_valid_pair(
    source: str,
    target: str,
    min_characters: int = 1,
    max_characters: int = 512,
    min_length_ratio: float = 1.0 / 3.0,
    max_length_ratio: float = 3.0,
    min_letter_ratio: float = 0.5,
) -> bool:
    """Cheap character-level quality filter."""
    source_length = len(source)
    target_length = len(target)
    if not (min_characters <= source_length <= max_characters):
        return False
    if not (min_characters <= target_length <= max_characters):
        return False
    ratio = source_length / target_length
    if not (min_length_ratio <= ratio <= max_length_ratio):
        return False
    # Reject lines that are mostly non-alphabetic noise (e.g. broken markup).
    for text in (source, target):
        letters = sum(character.isalpha() for character in text)
        if letters / len(text) < min_letter_ratio:
            return False
    return True


def pair_fingerprint(source: str, target: str) -> int:
    """Stable 32-bit fingerprint used for deduplication and dev splitting."""
    key = f"{source}\x00{target}".encode()
    return zlib.crc32(key)


# ----------------------------------------------------------------------
# Corpus preparation
# ----------------------------------------------------------------------
@dataclass
class PrepareConfig:
    out_dir: str = "data/processed"
    tokenizer_path: str = "artifacts/tokenizer/tokenizer.model"
    source_name: str = "corpus"
    max_source_length: int = 128
    max_target_length: int = 128
    min_characters: int = 1
    max_characters: int = 512
    min_length_ratio: float = 1.0 / 3.0
    max_length_ratio: float = 3.0
    # Uniform hash-based dev split; 0.0007 means roughly 0.07% of the data.
    dev_fraction: float = 0.0
    seed: int = 42
    num_workers: int = 8
    max_pairs: int = 0  # 0 = no limit (useful for smoke tests)
    chunk_size: int = 8192

    @property
    def tokenizer_meta_path(self) -> Path:
        return Path(self.out_dir) / "meta.json"


_WORKER_STATE: dict[str, object] = {}


def _init_worker(
    tokenizer_path: str, max_source_length: int, max_target_length: int
) -> None:
    _WORKER_STATE["tokenizer"] = MTTokenizer(tokenizer_path)
    _WORKER_STATE["max_source_length"] = max_source_length
    _WORKER_STATE["max_target_length"] = max_target_length


def _encode_pairs(
    pairs: Sequence[tuple[str, str]],
) -> list[tuple[list[int], list[int]] | None]:
    """Tokenize a chunk of pairs; ``None`` marks pairs that got dropped."""
    tokenizer: MTTokenizer = _WORKER_STATE["tokenizer"]  # type: ignore[assignment]
    max_source_length = int(_WORKER_STATE["max_source_length"])  # type: ignore[arg-type]
    max_target_length = int(_WORKER_STATE["max_target_length"])  # type: ignore[arg-type]
    encoded: list[tuple[list[int], list[int]] | None] = []
    for source, target in pairs:
        source_ids = tokenizer.encode(source)
        target_ids = tokenizer.encode(target)
        if not source_ids or not target_ids:
            encoded.append(None)
        elif (
            len(source_ids) > max_source_length
            or len(target_ids) > max_target_length
        ):
            encoded.append(None)
        else:
            encoded.append((source_ids, target_ids))
    return encoded


class _BinWriter:
    """Append-only writer for one (tokens, offsets) array pair."""

    def __init__(self, tokens_path: Path, offsets_path: Path) -> None:
        self.tokens_path = tokens_path
        self.offsets_path = offsets_path
        self._tokens = open(tokens_path, "wb")
        self.offsets = [0]
        self.num_tokens = 0

    def write(self, token_ids: Sequence[int]) -> None:
        array = np.asarray(token_ids, dtype=np.uint16)
        array.tofile(self._tokens)
        self.num_tokens += array.size
        self.offsets.append(self.num_tokens)

    def close(self) -> None:
        self._tokens.close()
        np.asarray(self.offsets, dtype=np.int64).tofile(self.offsets_path)


def _filtered_chunks(
    pairs,
    config: PrepareConfig,
    seen: set,
    rejected: dict[str, int],
):
    """Yield chunks of normalized, filtered, deduplicated triples."""
    chunk: list[tuple[str, str, str, int]] = []
    processed = 0
    for source, target, origin in pairs:
        if config.max_pairs and processed >= config.max_pairs:
            break
        processed += 1
        source = normalize_text(source)
        target = normalize_text(target)
        if (
            len(source) > config.max_characters
            or len(target) > config.max_characters
        ):
            rejected["length"] += 1
            continue
        if not is_valid_pair(
            source,
            target,
            min_characters=config.min_characters,
            max_characters=config.max_characters,
            min_length_ratio=config.min_length_ratio,
            max_length_ratio=config.max_length_ratio,
        ):
            rejected["quality"] += 1
            continue
        fingerprint = pair_fingerprint(source, target)
        if fingerprint in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(fingerprint)
        chunk.append((source, target, origin, fingerprint))
        if len(chunk) >= config.chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def prepare_corpus(
    pairs,
    config: PrepareConfig,
    log_every: int = 200000,
) -> dict:
    """Filter, deduplicate, split and tokenize a parallel corpus.

    ``pairs`` yields ``(source, target, origin)`` triples in the raw
    orientation (source = German, target = English).

    Returns the metadata dictionary that is also written to ``meta.json``.
    """
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = MTTokenizer(config.tokenizer_path)
    dev_threshold = int(round(config.dev_fraction * DEV_MODULUS))

    writers = {
        "train": (
            _BinWriter(
                out_dir / "train.src.bin", out_dir / "train.src.offsets"
            ),
            _BinWriter(
                out_dir / "train.tgt.bin", out_dir / "train.tgt.offsets"
            ),
        ),
        "valid": (
            _BinWriter(
                out_dir / "valid.src.bin", out_dir / "valid.src.offsets"
            ),
            _BinWriter(
                out_dir / "valid.tgt.bin", out_dir / "valid.tgt.offsets"
            ),
        ),
    }
    stats = {
        "train": {"pairs": 0, "src_tokens": 0, "tgt_tokens": 0},
        "valid": {"pairs": 0, "src_tokens": 0, "tgt_tokens": 0},
    }
    origin_counts: dict[str, int] = {}
    seen: set = set()
    rejected = {
        "length": 0,
        "quality": 0,
        "duplicate": 0,
        "tokenization": 0,
    }
    processed = 0

    def handle(
        chunk: Sequence[tuple[str, str, str, int]],
        encoded: Sequence[tuple[list[int], list[int]] | None],
    ) -> None:
        nonlocal processed
        for (_, _, origin, fingerprint), token_ids in zip(
            chunk, encoded, strict=True
        ):
            if token_ids is None:
                rejected["tokenization"] += 1
                continue
            use_dev = dev_threshold > 0 and (
                fingerprint % DEV_MODULUS < dev_threshold
            )
            split = "valid" if use_dev else "train"
            source_ids, target_ids = token_ids
            source_writer, target_writer = writers[split]
            source_writer.write(source_ids)
            target_writer.write(target_ids)
            stats[split]["pairs"] += 1
            stats[split]["src_tokens"] += len(source_ids)
            stats[split]["tgt_tokens"] += len(target_ids)
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
        processed += len(chunk)
        if log_every and processed % log_every < config.chunk_size:
            print(
                f"  processed {processed:,} raw pairs "
                f"(train={stats['train']['pairs']:,}, "
                f"valid={stats['valid']['pairs']:,})",
                flush=True,
            )

    pool = mp.Pool(
        processes=max(1, config.num_workers),
        initializer=_init_worker,
        initargs=(
            config.tokenizer_path,
            config.max_source_length,
            config.max_target_length,
        ),
    )
    in_flight: deque = deque()
    max_in_flight = max(2, 2 * config.num_workers)
    try:
        for chunk in _filtered_chunks(pairs, config, seen, rejected):
            pairs_only = [(source, target) for source, target, _, _ in chunk]
            in_flight.append(
                (chunk, pool.apply_async(_encode_pairs, (pairs_only,)))
            )
            while len(in_flight) >= max_in_flight:
                head_chunk, result = in_flight.popleft()
                handle(head_chunk, result.get())
        while in_flight:
            head_chunk, result = in_flight.popleft()
            handle(head_chunk, result.get())
    finally:
        pool.close()
        pool.join()
        for split in writers:
            for writer in writers[split]:
                writer.close()

    meta = {
        "name": config.source_name,
        "orientation": {"source": "de", "target": "en"},
        "directions": ["de-en", "en-de"],
        "num_train_pairs": stats["train"]["pairs"],
        "num_valid_pairs": stats["valid"]["pairs"],
        "num_train_src_tokens": stats["train"]["src_tokens"],
        "num_train_tgt_tokens": stats["train"]["tgt_tokens"],
        "origins": origin_counts,
        "rejected": rejected,
        "filters": {
            "min_characters": config.min_characters,
            "max_characters": config.max_characters,
            "min_length_ratio": config.min_length_ratio,
            "max_length_ratio": config.max_length_ratio,
            "max_source_length": config.max_source_length,
            "max_target_length": config.max_target_length,
            "dev_fraction": config.dev_fraction,
        },
        "tokenizer": {
            "path": str(config.tokenizer_path),
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
            "unk_id": tokenizer.unk_id,
            "eos_id": tokenizer.eos_id,
            "lang_token_ids": dict(tokenizer.lang_token_ids),
        },
        "seed": config.seed,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    return meta


# ----------------------------------------------------------------------
# Reading and batching
# ----------------------------------------------------------------------
class TokenizedCorpus:
    """Memory-mapped access to one processed split."""

    def __init__(self, directory: str | Path, split: str = "train") -> None:
        self.directory = Path(directory)
        self.split = split
        self.source = np.memmap(
            self.directory / f"{split}.src.bin", dtype=np.uint16, mode="r"
        )
        self.target = np.memmap(
            self.directory / f"{split}.tgt.bin", dtype=np.uint16, mode="r"
        )
        self.source_offsets = np.fromfile(
            self.directory / f"{split}.src.offsets", dtype=np.int64
        )
        self.target_offsets = np.fromfile(
            self.directory / f"{split}.tgt.offsets", dtype=np.int64
        )
        if len(self.source_offsets) != len(self.target_offsets):
            raise ValueError(
                f"source/target offsets length mismatch in {self.directory}"
            )

    def __len__(self) -> int:
        return len(self.source_offsets) - 1

    def source_tokens(self, index: int) -> np.ndarray:
        start, end = self.source_offsets[index], self.source_offsets[index + 1]
        return np.asarray(self.source[start:end], dtype=np.int64)

    def target_tokens(self, index: int) -> np.ndarray:
        start, end = self.target_offsets[index], self.target_offsets[index + 1]
        return np.asarray(self.target[start:end], dtype=np.int64)

    def lengths(self) -> tuple[np.ndarray, np.ndarray]:
        source_lengths = np.diff(self.source_offsets)
        target_lengths = np.diff(self.target_offsets)
        return source_lengths, target_lengths


@dataclass
class Batch:
    source: torch.Tensor
    target_input: torch.Tensor
    target_output: torch.Tensor
    source_padding_mask: torch.Tensor
    target_padding_mask: torch.Tensor
    direction: str
    num_target_tokens: int = 0


def _make_batch(
    indices: Sequence[int],
    direction: str,
    corpus: TokenizedCorpus,
    tokenizer_meta: dict,
    pad_id: int,
    eos_id: int,
) -> Batch:
    sources = [corpus.source_tokens(index) for index in indices]
    targets = [corpus.target_tokens(index) for index in indices]
    if direction == "en-de":
        sources, targets = targets, sources
        target_language = "de"
    else:
        target_language = "en"

    language_token = tokenizer_meta["tokenizer"]["lang_token_ids"][
        target_language
    ]
    max_source = max(len(item) for item in sources) + 1
    max_target = max(len(item) for item in targets) + 1

    source_batch = np.full((len(indices), max_source), pad_id, dtype=np.int64)
    target_input = np.full((len(indices), max_target), pad_id, dtype=np.int64)
    target_output = np.full((len(indices), max_target), pad_id, dtype=np.int64)
    for row, (source, target) in enumerate(zip(sources, targets, strict=True)):
        source_batch[row, : len(source)] = source
        source_batch[row, len(source)] = eos_id
        target_input[row, 0] = language_token
        target_input[row, 1 : len(target) + 1] = target
        target_output[row, : len(target)] = target
        target_output[row, len(target)] = eos_id

    return Batch(
        source=torch.from_numpy(source_batch),
        target_input=torch.from_numpy(target_input),
        target_output=torch.from_numpy(target_output),
        source_padding_mask=torch.from_numpy(source_batch == pad_id),
        target_padding_mask=torch.from_numpy(target_input == pad_id),
        direction=direction,
        num_target_tokens=int((target_output != pad_id).sum()),
    )


def iter_batches(
    corpus: TokenizedCorpus,
    tokenizer_meta: dict,
    *,
    direction: str = "both",
    max_tokens: int = 16384,
    max_batch_size: int = 256,
    shuffle: bool = True,
    sort_within_chunk: bool = True,
    chunk_size: int = 10000,
    seed: int = 42,
    epoch: int = 0,
) -> Iterator[Batch]:
    """Yield token batches sorted within shuffled chunks.

    With ``direction="both"`` each pair is used in one direction per epoch;
    the direction alternates across epochs so every pair is eventually seen
    in both directions.
    """
    pad_id = tokenizer_meta["tokenizer"]["pad_id"]
    eos_id = tokenizer_meta["tokenizer"]["eos_id"]
    source_lengths, target_lengths = corpus.lengths()
    combined_lengths = source_lengths + target_lengths

    order = np.arange(len(corpus))
    rng = np.random.default_rng(seed + epoch)
    if shuffle:
        rng.shuffle(order)

    for chunk_start in range(0, len(order), chunk_size):
        chunk = order[chunk_start : chunk_start + chunk_size]
        if direction == "both":
            groups: dict[str, list[int]] = {"de-en": [], "en-de": []}
            for index in chunk:
                key = "de-en" if (int(index) + epoch) % 2 == 0 else "en-de"
                groups[key].append(int(index))
        else:
            groups = {direction: [int(index) for index in chunk]}

        batch_index_lists: list[tuple[str, list[int]]] = []
        for group_direction, group_indices in groups.items():
            if not group_indices:
                continue
            if sort_within_chunk:
                group_indices.sort(key=lambda index: combined_lengths[index])
            current: list[int] = []
            current_tokens = 0
            for index in group_indices:
                length = int(combined_lengths[index])
                if current and (
                    current_tokens + length > max_tokens
                    or len(current) >= max_batch_size
                ):
                    batch_index_lists.append((group_direction, current))
                    current = []
                    current_tokens = 0
                current.append(index)
                current_tokens += length
            if current:
                batch_index_lists.append((group_direction, current))

        if shuffle:
            rng.shuffle(batch_index_lists)
        for group_direction, indices in batch_index_lists:
            yield _make_batch(
                indices,
                group_direction,
                corpus,
                tokenizer_meta,
                pad_id,
                eos_id,
            )


def load_meta(data_dir: str | Path) -> dict:
    with open(Path(data_dir) / "meta.json", encoding="utf-8") as handle:
        return json.load(handle)
