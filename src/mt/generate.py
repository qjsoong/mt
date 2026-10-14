"""Inference: greedy and beam-search translation with KV caching."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ModelConfig
from .model import KVCache, Transformer
from .tokenizer import MTTokenizer
from .utils import resolve_device


def _expand_cache(cache: list[KVCache], beam_size: int) -> list[KVCache]:
    return [
        (
            key.repeat_interleave(beam_size, dim=0),
            value.repeat_interleave(beam_size, dim=0),
        )
        for key, value in cache
    ]


def _reorder_cache(cache: list[KVCache], index: torch.Tensor) -> list[KVCache]:
    return [
        (key.index_select(0, index), value.index_select(0, index))
        for key, value in cache
    ]


def _length_penalty(lengths: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha == 0:
        return torch.ones_like(lengths, dtype=torch.float32)
    return ((5.0 + lengths.float()) / 6.0) ** alpha


@torch.no_grad()
def _beam_search(
    model: Transformer,
    source_tokens: torch.Tensor,
    source_padding_mask: torch.Tensor,
    bos_token: int,
    eos_token: int,
    pad_token: int,
    beam_size: int,
    max_length: int,
    length_penalty: float,
) -> torch.Tensor:
    """Beam search for one padded batch.

    Returns a (batch, beam_size, max_length) tensor of token ids sorted by
    score (best hypothesis first). Sequences are right-padded with
    ``pad_token``.

    Finished hypotheses are kept in a top-k buffer, independently of the
    live beam slots. A live slot may therefore be reused for a different
    hypothesis after its previous one emitted EOS without losing it.
    """
    device = source_tokens.device
    batch_size = source_tokens.size(0)
    vocab_size = model.cfg.vocab_size
    memory = model.encode(source_tokens, source_padding_mask)
    cross_cache = model.prepare_cross_cache(memory)

    final_scores = torch.full((batch_size, beam_size), -1e30, device=device)
    final_sequences = torch.full(
        (batch_size, beam_size, max_length),
        pad_token,
        dtype=torch.long,
        device=device,
    )

    def finalize(mask: torch.Tensor, current_scores: torch.Tensor) -> None:
        """Merge the selected hypotheses into the top-k finished buffer."""
        nonlocal final_scores, final_sequences
        counts = mask.sum(dim=1)
        width = int(counts.max())
        if width == 0:
            return
        normalized = current_scores / _length_penalty(lengths, length_penalty)
        candidate_scores = torch.full((batch_size, width), -1e30, device=device)
        candidate_sequences = torch.full(
            (batch_size, width, max_length),
            pad_token,
            dtype=torch.long,
            device=device,
        )
        ranks = (mask.cumsum(dim=1) - 1).clamp(min=0)
        rows = torch.arange(batch_size, device=device)[:, None].expand_as(mask)
        batch_index = rows[mask]
        slot_index = ranks[mask]
        candidate_scores[batch_index, slot_index] = normalized[mask]
        candidate_sequences[batch_index, slot_index, : sequences.size(2)] = (
            sequences[mask]
        )

        merged_scores = torch.cat([final_scores, candidate_scores], dim=1)
        merged_sequences = torch.cat(
            [final_sequences, candidate_sequences], dim=1
        )
        top_scores, top_indices = merged_scores.topk(beam_size, dim=1)
        final_scores = top_scores
        final_sequences = merged_sequences.gather(
            1, top_indices.unsqueeze(-1).expand(-1, -1, max_length)
        )

    # --- First step: expand the single BOS hypothesis per sequence. -------
    # ``beam_size`` can exceed the vocabulary in tests; cap the initial
    # expansion and pad the remaining slots with dead hypotheses.
    initial_beam = min(beam_size, vocab_size)
    first = torch.full(
        (batch_size, 1), bos_token, dtype=torch.long, device=device
    )
    logits, cache = model.decode_step(
        first, memory, source_padding_mask, cross_cache, None, 0
    )
    log_probs = F.log_softmax(logits.float(), dim=-1)
    scores, tokens = log_probs.topk(initial_beam, dim=-1)  # (B, beam)
    sequences = tokens.unsqueeze(-1)  # (B, beam, 1)
    lengths = torch.ones(
        (batch_size, initial_beam), dtype=torch.long, device=device
    )
    beam_source = torch.zeros(
        (batch_size, initial_beam), dtype=torch.long, device=device
    )
    alive = torch.ones(
        (batch_size, initial_beam), dtype=torch.bool, device=device
    )
    if initial_beam < beam_size:
        padding = beam_size - initial_beam
        scores = F.pad(scores, (0, padding), value=-1e30)
        tokens = F.pad(tokens, (0, padding), value=pad_token)
        sequences = F.pad(sequences, (0, 0, 0, padding), value=pad_token)
        lengths = F.pad(lengths, (0, padding), value=0)
        beam_source = F.pad(beam_source, (0, padding), value=0)
        alive = F.pad(alive, (0, padding), value=False)

    first_eos = tokens == eos_token
    finalize(first_eos, scores)
    alive = alive & ~first_eos
    scores = scores.masked_fill(~alive, -1e30)
    # From here on the decoder batch is (batch * beam); expand encoder-side
    # caches and the self-attention cache accordingly.
    cache = _expand_cache(cache, beam_size)
    cross_cache = _expand_cache(cross_cache, beam_size)
    cache = _reorder_cache(cache, beam_source.reshape(-1))

    for step in range(1, max_length):
        if not alive.any():
            break
        current = sequences[:, :, -1].reshape(batch_size * beam_size, 1)
        flat_beam_source = (
            torch.arange(batch_size, device=device)[:, None] * beam_size
            + beam_source
        ).reshape(-1)
        cache = _reorder_cache(cache, flat_beam_source)
        logits, cache = model.decode_step(
            current, memory, source_padding_mask, cross_cache, cache, step
        )
        log_probs = F.log_softmax(logits.float(), dim=-1).view(
            batch_size, beam_size, vocab_size
        )
        candidate_scores = scores.unsqueeze(-1) + log_probs
        candidate_scores = candidate_scores.view(
            batch_size, beam_size * vocab_size
        )
        top_scores, top_indices = candidate_scores.topk(beam_size, dim=-1)
        beam_source = top_indices // vocab_size
        tokens = top_indices % vocab_size

        sequences = sequences.gather(
            1,
            beam_source.unsqueeze(-1).expand(-1, -1, sequences.size(-1)),
        )
        lengths = lengths.gather(1, beam_source) + 1
        sequences = torch.cat([sequences, tokens.unsqueeze(-1)], dim=2)

        emitted_eos = tokens == eos_token
        finalize(emitted_eos, top_scores)
        alive = alive.gather(1, beam_source) & ~emitted_eos
        scores = top_scores.masked_fill(~alive, -1e30)

    # Add hypotheses that are still running when ``max_length`` is reached
    # and return the best ones sorted by score.
    finalize(alive, scores)
    return final_sequences


class Translator:
    """Batch translator around :class:`~mt.model.Transformer`."""

    def __init__(
        self,
        model: Transformer,
        tokenizer: MTTokenizer,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path | None = None,
        device: str = "auto",
    ) -> Translator:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model_config = ModelConfig(**checkpoint["model_config"])
        model = Transformer(model_config)
        model.load_state_dict(checkpoint["model"])
        tokenizer_path = tokenizer_path or checkpoint.get("tokenizer_path")
        if tokenizer_path is None:
            raise ValueError(
                "no tokenizer path given and none stored in checkpoint"
            )
        tokenizer = MTTokenizer(tokenizer_path)
        if tokenizer.vocab_size != model_config.vocab_size:
            raise ValueError(
                f"tokenizer vocab size {tokenizer.vocab_size} does not match "
                f"model vocab size {model_config.vocab_size}"
            )
        return cls(model, tokenizer, device=device)

    # ------------------------------------------------------------------
    def _encode_texts(
        self, texts: Sequence[str], max_source_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded: list[list[int]] = []
        for text in texts:
            ids = self.tokenizer.encode(text)
            if len(ids) >= max_source_length:
                ids = ids[: max_source_length - 1]
            encoded.append(ids + [self.tokenizer.eos_id])
        width = max(len(item) for item in encoded)
        tokens = torch.full(
            (len(encoded), width), self.tokenizer.pad_id, dtype=torch.long
        )
        for row, ids in enumerate(encoded):
            tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask = tokens == self.tokenizer.pad_id
        return tokens.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def translate(
        self,
        texts: Sequence[str],
        direction: str,
        beam_size: int = 5,
        length_penalty: float = 0.6,
        max_length: int | None = None,
        batch_size: int = 64,
        max_source_length: int = 512,
    ) -> list[str]:
        """Translate ``texts`` from the source to the target language.

        ``direction`` is ``"de-en"`` or ``"en-de"``.
        """
        if direction not in {"de-en", "en-de"}:
            raise ValueError(f"unknown direction: {direction}")
        target_language = direction.split("-")[1]
        bos_token = self.tokenizer.target_language_token(target_language)
        results: list[str | None] = [None] * len(texts)

        # Sort by length so that batches contain similarly sized sentences.
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            batch_texts = [texts[index] for index in chunk]
            tokens, mask = self._encode_texts(batch_texts, max_source_length)
            if max_length is not None:
                limit = max_length
            else:
                limit = int(tokens.size(1) * 1.5) + 20
                limit = min(limit, self.model.cfg.max_positions - 1)
            if beam_size <= 1:
                sequences = _beam_search(
                    self.model,
                    tokens,
                    mask,
                    bos_token=bos_token,
                    eos_token=self.tokenizer.eos_id,
                    pad_token=self.tokenizer.pad_id,
                    beam_size=1,
                    max_length=limit,
                    length_penalty=0.0,
                )
            else:
                sequences = _beam_search(
                    self.model,
                    tokens,
                    mask,
                    bos_token=bos_token,
                    eos_token=self.tokenizer.eos_id,
                    pad_token=self.tokenizer.pad_id,
                    beam_size=beam_size,
                    max_length=limit,
                    length_penalty=length_penalty,
                )
            best = sequences[:, 0]
            for row, index in enumerate(chunk):
                results[index] = self.tokenizer.decode(best[row].tolist())
        return [text if text is not None else "" for text in results]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate text with mt")
    parser.add_argument("--model", required=True, help="checkpoint path")
    parser.add_argument(
        "--tokenizer", default=None, help="tokenizer.model path"
    )
    parser.add_argument(
        "--direction", required=True, choices=["de-en", "en-de"]
    )
    parser.add_argument(
        "--input", default=None, help="input file, one sentence per line"
    )
    parser.add_argument("--output", default=None, help="output file")
    parser.add_argument(
        "--text", default=None, help="translate a single string"
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args(argv)

    translator = Translator.from_checkpoint(
        arguments.model, arguments.tokenizer, device=arguments.device
    )
    if arguments.text is not None:
        lines = [arguments.text]
    elif arguments.input is not None:
        with open(arguments.input, encoding="utf-8") as handle:
            lines = [line.rstrip("\n") for line in handle]
    else:
        lines = [line.rstrip("\n") for line in sys.stdin]

    translations = translator.translate(
        lines,
        direction=arguments.direction,
        beam_size=arguments.beam_size,
        length_penalty=arguments.length_penalty,
        batch_size=arguments.batch_size,
    )
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as handle:
            for line in translations:
                handle.write(line + "\n")
    else:
        for line in translations:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
