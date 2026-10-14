"""SentencePiece tokenizer wrapper.

A single shared subword vocabulary is trained on both languages. Special
tokens follow the Marian/NLLB conventions::

    <pad> = 0, <unk> = 1, <s> = 2, </s> = 3

The target language of the decoder is selected with a language token that is
prepended to the decoder input: ``<2de>`` or ``<2en>``.
"""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

LANG_TOKENS: dict[str, str] = {"de": "<2de>", "en": "<2en>"}


class MTTokenizer:
    """Thin wrapper around :class:`sentencepiece.SentencePieceProcessor`."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.lang_token_ids: dict[str, int] = {}
        for language, piece in LANG_TOKENS.items():
            token_id = self.sp.piece_to_id(piece)
            if token_id == self.sp.unk_id():
                raise ValueError(
                    f"language token {piece!r} is missing from "
                    f"{self.model_path}"
                )
            self.lang_token_ids[language] = token_id

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids (no special tokens added)."""
        return self.sp.encode(text, out_type=int)

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = True
    ) -> str:
        if skip_special_tokens:
            token_ids = [
                token_id
                for token_id in token_ids
                if token_id not in (PAD_ID, BOS_ID, EOS_ID)
                and token_id not in self.lang_token_ids.values()
            ]
        return self.sp.decode(token_ids)

    def target_language_token(self, language: str) -> int:
        if language not in self.lang_token_ids:
            raise KeyError(f"unknown target language: {language}")
        return self.lang_token_ids[language]

    def save_meta(self, path: str | Path) -> None:
        meta = {
            "model_path": str(self.model_path),
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "bos_id": BOS_ID,
            "eos_id": self.eos_id,
            "lang_token_ids": dict(self.lang_token_ids),
            "lang_tokens": dict(LANG_TOKENS),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
            handle.write("\n")


def load_tokenizer_meta(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def train_tokenizer(
    corpus_files: list[str | Path],
    output_prefix: str | Path,
    vocab_size: int = 32000,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    num_threads: int = 16,
    max_sentences: int | None = None,
) -> Path:
    """Train a SentencePiece model on the given plain-text files.

    Returns the path to the produced ``.model`` file.
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input=",".join(str(p) for p in corpus_files),
        model_prefix=str(output_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        user_defined_symbols=list(LANG_TOKENS.values()),
        byte_fallback=True,
        num_threads=num_threads,
        input_sentence_size=max_sentences or 0,
        shuffle_input_sentence=bool(max_sentences),
    )
    spm.SentencePieceTrainer.train(**kwargs)
    return output_prefix.with_suffix(".model")
