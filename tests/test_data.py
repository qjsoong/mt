"""Data pipeline tests: preparation, direction handling, batching."""

from __future__ import annotations

from pathlib import Path

from mt.data import (
    PrepareConfig,
    TokenizedCorpus,
    iter_batches,
    pair_fingerprint,
    prepare_corpus,
)
from mt.tokenizer import MTTokenizer

GERMAN = [
    "Der Hund schläft auf dem Sofa.",
    "Wir gehen morgen in die Stadt.",
    "Das Wetter ist heute sehr schön.",
    "Ich lese gerade ein spannendes Buch.",
    "Bitte schließen Sie die Tür leise.",
]
ENGLISH = [
    "The dog is sleeping on the sofa.",
    "We are going to the city tomorrow.",
    "The weather is very nice today.",
    "I am reading an exciting book right now.",
    "Please close the door quietly.",
]


def build_corpus(tokenizer_dir: Path, tmp_path: Path):
    pairs = [
        (german, english, "toy")
        for german, english in zip(GERMAN * 4, ENGLISH * 4, strict=True)
    ]
    config = PrepareConfig(
        out_dir=str(tmp_path / "processed"),
        tokenizer_path=str(tokenizer_dir / "tokenizer.model"),
        dev_fraction=0.3,
        num_workers=1,
        max_source_length=32,
        max_target_length=32,
    )
    meta = prepare_corpus(iter(pairs), config)
    return meta, tmp_path / "processed"


def test_prepare_splits_and_filters(
    tokenizer_dir: Path, tmp_path: Path
) -> None:
    meta, _ = build_corpus(tokenizer_dir, tmp_path)
    assert meta["num_train_pairs"] + meta["num_valid_pairs"] > 0
    # Duplicate sentences are collapsed.
    assert meta["rejected"]["duplicate"] > 0


def test_corpus_shapes(tokenizer_dir: Path, tmp_path: Path) -> None:
    meta, directory = build_corpus(tokenizer_dir, tmp_path)
    corpus = TokenizedCorpus(directory, "train")
    assert len(corpus) == meta["num_train_pairs"]
    source, target = corpus.source_tokens(0), corpus.target_tokens(0)
    assert len(source) > 0 and len(target) > 0


def test_batch_construction(tokenizer_dir: Path, tmp_path: Path) -> None:
    meta, directory = build_corpus(tokenizer_dir, tmp_path)
    corpus = TokenizedCorpus(directory, "train")
    tokenizer = MTTokenizer(tokenizer_dir / "tokenizer.model")
    batches = list(
        iter_batches(
            corpus,
            meta,
            direction="both",
            max_tokens=64,
            max_batch_size=4,
            shuffle=False,
            seed=1,
        )
    )
    assert batches
    pad_id = meta["tokenizer"]["pad_id"]
    eos_id = meta["tokenizer"]["eos_id"]
    for batch in batches:
        assert batch.source.size(0) == batch.target_input.size(0)
        assert batch.source.size(1) == batch.source_padding_mask.size(1)
        assert batch.target_input.size(1) == batch.target_output.size(1)
        expected = tokenizer.lang_token_ids[batch.direction.split("-")[1]]
        assert (batch.target_input[:, 0] == expected).all()
        # Non-pad source rows end with EOS.
        lengths = (~batch.source_padding_mask).sum(dim=1)
        for row, length in enumerate(lengths):
            assert batch.source[row, length - 1] == eos_id
        # Padding masks match the pad token.
        assert (batch.source[batch.source_padding_mask] == pad_id).all()
        assert (batch.target_input[batch.target_padding_mask] == pad_id).all()
        assert batch.num_target_tokens == int(
            (batch.target_output != pad_id).sum()
        )


def test_direction_selects_correct_side(
    tokenizer_dir: Path, tmp_path: Path
) -> None:
    meta, directory = build_corpus(tokenizer_dir, tmp_path)
    corpus = TokenizedCorpus(directory, "train")
    for batch in iter_batches(
        corpus, meta, direction="en-de", max_tokens=512, shuffle=False, seed=2
    ):
        # Every source row must be one of the stored English sentences.
        stored = {
            tuple(corpus.target_tokens(index).tolist())
            for index in range(len(corpus))
        }
        for row in range(batch.source.size(0)):
            length = int((~batch.source_padding_mask[row]).sum())
            tokens = tuple(batch.source[row, : length - 1].tolist())
            assert tokens in stored
        break


def test_fingerprint_is_stable() -> None:
    assert pair_fingerprint("a", "b") == pair_fingerprint("a", "b")
    assert pair_fingerprint("a", "b") != pair_fingerprint("b", "a")
