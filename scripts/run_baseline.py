#!/usr/bin/env python3
"""Evaluate a Hugging Face translation baseline on the same test sets.

Example::

    python scripts/run_baseline.py \
        --model-name Helsinki-NLP/opus-mt-de-en \
        --direction de-en \
        --test-sets wmt14,wmt19 \
        --output results/opus-mt-de-en.json

For multilingual checkpoints (NLLB, M2M-100, mBART) pass language codes::

    python scripts/run_baseline.py \
        --model-name facebook/nllb-200-distilled-600M \
        --direction de-en \
        --source-lang-code deu_Latn --target-lang-code eng_Latn \
        --output results/nllb-600m.de-en.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mt.evaluate import load_test_set, score  # noqa: E402
from mt.utils import resolve_device  # noqa: E402


def translate_batch(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    beam_size: int,
    batch_size: int,
    max_length: int,
    forced_bos_token_id: int | None,
) -> list[str]:
    hypotheses: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                num_beams=beam_size,
                max_length=max_length,
                forced_bos_token_id=forced_bos_token_id,
            )
        hypotheses.extend(
            tokenizer.batch_decode(generated, skip_special_tokens=True)
        )
    return hypotheses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--direction", required=True, choices=["de-en", "en-de"]
    )
    parser.add_argument("--test-sets", default="wmt14,wmt19")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--source-lang-code", default=None)
    parser.add_argument("--target-lang-code", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hypotheses-dir", default=None)
    arguments = parser.parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = resolve_device(arguments.device)
    print(f"loading {arguments.model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(arguments.model_name)
    if arguments.source_lang_code:
        tokenizer.src_lang = arguments.source_lang_code
    model = AutoModelForSeq2SeqLM.from_pretrained(arguments.model_name).to(
        device
    )
    model.eval()
    forced_bos_token_id = None
    if arguments.target_lang_code:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(
            arguments.target_lang_code
        )
    parameters = sum(parameter.numel() for parameter in model.parameters())

    results = []
    for test_set in arguments.test_sets.split(","):
        test_set = test_set.strip()
        sources, references = load_test_set(test_set, arguments.direction)
        start = time.time()
        hypotheses = translate_batch(
            model,
            tokenizer,
            sources,
            device,
            arguments.beam_size,
            arguments.batch_size,
            arguments.max_length,
            forced_bos_token_id,
        )
        elapsed = time.time() - start
        metrics = score(hypotheses, references)
        if arguments.hypotheses_dir:
            path = (
                Path(arguments.hypotheses_dir)
                / f"{test_set}.{arguments.direction}.hyp.txt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                for line in hypotheses:
                    handle.write(line + "\n")
        results.append(
            {
                "test_set": test_set,
                "direction": arguments.direction,
                "num_sentences": len(sources),
                "beam_size": arguments.beam_size,
                "bleu": metrics["bleu"],
                "chrf": metrics["chrf"],
                "seconds": elapsed,
                "sentences_per_second": len(sources) / max(1e-9, elapsed),
            }
        )
        print(
            f"{test_set} {arguments.direction}: BLEU {metrics['bleu']:.2f} | "
            f"chrF {metrics['chrf']:.2f} | {len(sources) / elapsed:.1f} sent/s"
        )

    payload = {
        "model": arguments.model_name,
        "parameters": parameters,
        "beam_size": arguments.beam_size,
        "results": results,
    }
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
