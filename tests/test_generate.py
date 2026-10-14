"""Generation tests: greedy/beam search behaviour and API plumbing."""

from __future__ import annotations

import itertools
import math

import torch

from mt.config import ModelConfig
from mt.generate import Translator, _beam_search
from mt.model import Transformer
from mt.tokenizer import MTTokenizer


def tiny_config(vocab_size: int = 12) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
        attention_dropout=0.0,
    )


def make_model(vocab_size: int = 12) -> Transformer:
    torch.manual_seed(0)
    return Transformer(tiny_config(vocab_size)).eval()


def test_beam_search_shapes() -> None:
    model = make_model()
    source = torch.randint(4, 12, (2, 5))
    mask = torch.zeros_like(source, dtype=torch.bool)
    sequences = _beam_search(
        model,
        source,
        mask,
        bos_token=2,
        eos_token=3,
        pad_token=0,
        beam_size=4,
        max_length=6,
        length_penalty=0.6,
    )
    assert sequences.shape == (2, 4, 6)
    assert sequences.dtype == torch.long


def test_greedy_matches_manual_argmax() -> None:
    model = make_model()
    source = torch.randint(4, 12, (1, 4))
    mask = torch.zeros_like(source, dtype=torch.bool)
    sequences = _beam_search(
        model,
        source,
        mask,
        bos_token=2,
        eos_token=3,
        pad_token=0,
        beam_size=1,
        max_length=4,
        length_penalty=0.0,
    )
    # Manual greedy decoding with the incremental interface.
    with torch.no_grad():
        memory = model.encode(source, mask)
        cross_cache = model.prepare_cross_cache(memory)
        cache = None
        previous = 2
        expected: list[int] = []
        for step in range(4):
            logits, cache = model.decode_step(
                torch.tensor([[previous]], dtype=torch.long),
                memory,
                mask,
                cross_cache,
                cache,
                step,
            )
            previous = int(logits.argmax(dim=-1).item())
            expected.append(previous)
            if previous == 3:
                break
    assert sequences[0, 0, : len(expected)].tolist() == expected


def exhaustive_best(
    model: Transformer,
    source: torch.Tensor,
    bos: int,
    eos: int,
    max_length: int,
    alpha: float,
) -> list[int]:
    """Brute-force best sequence under length-normalized log probability.

    Enumerates every sequence the beam search could possibly return: a
    sequence ends either with EOS or at ``max_length``, and EOS is never
    followed by another token.
    """
    vocab = model.cfg.vocab_size
    best_sequence: list[int] = []
    best_score = -math.inf
    for length in range(1, max_length + 1):
        for candidate in itertools.product(range(vocab), repeat=length):
            if eos in candidate[:-1]:
                continue
            if candidate[-1] != eos and length < max_length:
                continue
            sequence = list(candidate)
            decoder_input = [bos] + sequence[:-1]
            with torch.no_grad():
                logits = model(
                    source, torch.tensor([decoder_input], dtype=torch.long)
                )
                log_probs = torch.log_softmax(logits[0], dim=-1)
            score = sum(
                float(log_probs[position, token])
                for position, token in enumerate(sequence)
            )
            penalty = ((5.0 + len(sequence)) / 6.0) ** alpha
            score /= penalty
            if score > best_score:
                best_score = score
                best_sequence = sequence
    return best_sequence


def test_beam_search_finds_optimal_sequence_with_large_beam() -> None:
    # A vocabulary of four tokens and two steps makes an exhaustive beam
    # (beam >= vocab ** max_length) equal to brute-force search.
    model = make_model(vocab_size=4)
    source = torch.tensor([[1, 2, 3]])
    mask = torch.zeros_like(source, dtype=torch.bool)
    max_length = 2
    sequences = _beam_search(
        model,
        source,
        mask,
        bos_token=2,
        eos_token=3,
        pad_token=0,
        beam_size=16,
        max_length=max_length,
        length_penalty=0.6,
    )
    best_sequence = exhaustive_best(
        model, source, bos=2, eos=3, max_length=max_length, alpha=0.6
    )
    produced = [token for token in sequences[0, 0].tolist() if token != 0]
    assert produced == best_sequence


def test_translator_returns_strings(tokenizer: MTTokenizer) -> None:
    model = make_model(vocab_size=tokenizer.vocab_size)
    translator = Translator(model, tokenizer, device="cpu")
    outputs = translator.translate(
        ["Der Hund schläft.", "Wir gehen morgen in die Stadt."],
        direction="de-en",
        beam_size=2,
        max_length=8,
        batch_size=2,
    )
    assert len(outputs) == 2
    assert all(isinstance(text, str) for text in outputs)
