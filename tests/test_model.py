"""Model correctness tests: shapes, masking, caching, tying."""

from __future__ import annotations

import pytest
import torch

from mt.config import ModelConfig
from mt.model import Transformer


def tiny_config(**overrides) -> ModelConfig:
    values = dict(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dropout=0.0,
        attention_dropout=0.0,
    )
    values.update(overrides)
    return ModelConfig(**values)


def random_batch(batch=2, source_length=5, target_length=4, vocab=64):
    source = torch.randint(4, vocab, (batch, source_length))
    target = torch.randint(4, vocab, (batch, target_length))
    return source, target


def test_forward_shape_and_loss() -> None:
    model = Transformer(tiny_config())
    source, target = random_batch()
    logits = model(source, target)
    assert logits.shape == (2, 4, 64)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 64), target.reshape(-1)
    )
    assert torch.isfinite(loss)


def test_embeddings_are_tied() -> None:
    model = Transformer(tiny_config(tie_embeddings=True))
    assert model.output_projection.weight is model.embedding.weight
    untied = Transformer(tiny_config(tie_embeddings=False))
    assert untied.output_projection.weight is not untied.embedding.weight


@pytest.mark.parametrize("norm", ["rmsnorm", "layernorm"])
@pytest.mark.parametrize("activation", ["swiglu", "gelu", "relu"])
@pytest.mark.parametrize("positional", ["sinusoidal", "learned"])
def test_architectures_run(norm: str, activation: str, positional: str) -> None:
    model = Transformer(
        tiny_config(norm=norm, activation=activation, positional=positional)
    )
    source, target = random_batch()
    logits = model(
        source, target, target_padding_mask=torch.zeros_like(target).bool()
    )
    assert logits.shape == (2, 4, 64)
    assert torch.isfinite(logits).all()


def test_source_padding_does_not_change_real_positions() -> None:
    model = Transformer(tiny_config()).eval()
    source, target = random_batch(batch=1, source_length=4)
    padded = torch.cat([source, torch.zeros(1, 3, dtype=torch.long)], dim=1)
    source_mask = torch.zeros_like(source, dtype=torch.bool)
    padded_mask = torch.zeros_like(padded, dtype=torch.bool)
    padded_mask[:, 4:] = True

    with torch.no_grad():
        plain = model.encode(source, source_mask)
        with_padding = model.encode(padded, padded_mask)
    torch.testing.assert_close(plain, with_padding[:, :4], atol=1e-5, rtol=1e-4)


def test_target_padding_does_not_change_real_positions() -> None:
    model = Transformer(tiny_config()).eval()
    source, target = random_batch(batch=1, target_length=3)
    padded_target = torch.cat(
        [target, torch.zeros(1, 2, dtype=torch.long)], dim=1
    )
    target_mask = torch.zeros_like(target, dtype=torch.bool)
    padded_mask = torch.zeros_like(padded_target, dtype=torch.bool)
    padded_mask[:, 3:] = True

    with torch.no_grad():
        first = model(source, target, target_padding_mask=target_mask)
        second = model(source, padded_target, target_padding_mask=padded_mask)
    torch.testing.assert_close(first, second[:, :3], atol=1e-5, rtol=1e-4)


def test_future_tokens_do_not_leak() -> None:
    model = Transformer(tiny_config()).eval()
    source, target = random_batch(batch=1, target_length=5)
    modified = target.clone()
    modified[0, -1] = (modified[0, -1] + 1) % 64
    with torch.no_grad():
        original = model(source, target)
        changed = model(source, modified)
    torch.testing.assert_close(
        original[:, :-1], changed[:, :-1], atol=1e-5, rtol=1e-4
    )


@pytest.mark.parametrize("positional", ["sinusoidal", "learned"])
def test_incremental_decoding_matches_full_forward(positional: str) -> None:
    model = Transformer(tiny_config(positional=positional)).eval()
    source, target = random_batch(batch=3, source_length=6, target_length=7)
    source_mask = torch.zeros_like(source, dtype=torch.bool)
    source_mask[0, 5:] = True

    with torch.no_grad():
        full_logits = model(source, target, source_padding_mask=source_mask)
        memory = model.encode(source, source_mask)
        cross_cache = model.prepare_cross_cache(memory)
        cache = None
        incremental_logits = []
        for position in range(target.size(1)):
            logits, cache = model.decode_step(
                target[:, position : position + 1],
                memory,
                source_mask,
                cross_cache,
                cache,
                position,
            )
            incremental_logits.append(logits)
    stacked = torch.stack(incremental_logits, dim=1)
    torch.testing.assert_close(full_logits, stacked, atol=1e-5, rtol=1e-4)


def test_tied_embeddings_reduce_parameter_count() -> None:
    tied = Transformer(tiny_config(tie_embeddings=True))
    untied = Transformer(tiny_config(tie_embeddings=False))
    assert tied.num_parameters() == sum(p.numel() for p in tied.parameters())
    assert tied.num_parameters() < untied.num_parameters()
