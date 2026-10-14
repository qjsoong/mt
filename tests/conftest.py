"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mt.tokenizer import MTTokenizer, train_tokenizer  # noqa: E402

SAMPLE_GERMAN = [
    "Der Hund schläft auf dem Sofa.",
    "Wir gehen morgen in die Stadt.",
    "Das Wetter ist heute sehr schön.",
    "Ich lese gerade ein spannendes Buch.",
    "Bitte schließen Sie die Tür leise.",
] * 4
SAMPLE_ENGLISH = [
    "The dog is sleeping on the sofa.",
    "We are going to the city tomorrow.",
    "The weather is very nice today.",
    "I am reading an exciting book right now.",
    "Please close the door quietly.",
] * 4


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("tokenizer")
    german_path = directory / "de.txt"
    english_path = directory / "en.txt"
    german_path.write_text("\n".join(SAMPLE_GERMAN) + "\n", encoding="utf-8")
    english_path.write_text("\n".join(SAMPLE_ENGLISH) + "\n", encoding="utf-8")
    model_path = train_tokenizer(
        [german_path, english_path],
        directory / "tokenizer",
        vocab_size=320,
        character_coverage=1.0,
        num_threads=2,
    )
    tokenizer = MTTokenizer(model_path)
    tokenizer.save_meta(directory / "meta.json")
    return directory


@pytest.fixture(scope="session")
def tokenizer(tokenizer_dir: Path) -> MTTokenizer:
    return MTTokenizer(tokenizer_dir / "tokenizer.model")
