"""Tokenizer tests (uses a tiny SentencePiece model from conftest)."""

from __future__ import annotations

from mt.tokenizer import MTTokenizer


def test_special_tokens(tokenizer: MTTokenizer) -> None:
    assert tokenizer.pad_id == 0
    assert tokenizer.unk_id == 1
    assert tokenizer.eos_id == 3
    assert set(tokenizer.lang_token_ids) == {"de", "en"}
    assert tokenizer.lang_token_ids["de"] != tokenizer.lang_token_ids["en"]


def test_roundtrip(tokenizer: MTTokenizer) -> None:
    for text in [
        "Der Hund schläft auf dem Sofa.",
        "The weather is very nice today.",
    ]:
        ids = tokenizer.encode(text)
        assert ids
        decoded = tokenizer.decode(ids)
        assert decoded.replace(" ", "") == text.replace(" ", "")


def test_language_tokens_are_skipped_when_decoding(
    tokenizer: MTTokenizer,
) -> None:
    german = tokenizer.encode("Der Hund schläft.")
    ids = [tokenizer.lang_token_ids["en"]] + german + [tokenizer.eos_id]
    decoded = tokenizer.decode(ids)
    assert "<2en>" not in decoded
    assert "Hund" in decoded


def test_vocab_size_matches_pieces(tokenizer: MTTokenizer) -> None:
    assert tokenizer.vocab_size == tokenizer.sp.get_piece_size()
