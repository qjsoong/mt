"""Evaluation on standard test sets.

Fetches test data with sacrebleu (so that tokenization matches published
numbers), translates with a checkpoint and reports BLEU and chrF.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from .generate import Translator

DEFAULT_TEST_SETS = ["wmt14", "wmt19"]


def load_test_set(
    test_set: str, direction: str
) -> tuple[list[str], list[list[str]]]:
    """Return (sources, references) for a sacrebleu test set.

    ``direction`` is ``"de-en"`` or ``"en-de"``.
    """
    from sacrebleu.utils import get_reference_files, get_source_file

    language_pair = direction.replace("-", "-")
    source_file = get_source_file(test_set, language_pair)
    reference_files = get_reference_files(test_set, language_pair)
    with open(source_file, encoding="utf-8") as handle:
        sources = [line.rstrip("\n") for line in handle]
    references: list[list[str]] = []
    for reference_file in reference_files:
        with open(reference_file, encoding="utf-8") as handle:
            references.append([line.rstrip("\n") for line in handle])
    return sources, references


def score(
    hypotheses: list[str], references: list[list[str]]
) -> dict[str, float]:
    from sacrebleu.metrics import BLEU, CHRF

    bleu = BLEU()
    chrf = CHRF()
    bleu_score = bleu.corpus_score(hypotheses, references)
    chrf_score = chrf.corpus_score(hypotheses, references)
    return {"bleu": bleu_score.score, "chrf": chrf_score.score}


def evaluate(
    translator: Translator,
    test_set: str,
    direction: str,
    beam_size: int = 5,
    batch_size: int = 64,
    max_sentences: int | None = None,
    save_hypotheses: Path | None = None,
) -> dict:
    sources, references = load_test_set(test_set, direction)
    if max_sentences:
        sources = sources[:max_sentences]
        references = [reference[:max_sentences] for reference in references]
    start = time.time()
    hypotheses = translator.translate(
        sources, direction=direction, beam_size=beam_size, batch_size=batch_size
    )
    elapsed = time.time() - start
    metrics = score(hypotheses, references)
    result = {
        "test_set": test_set,
        "direction": direction,
        "num_sentences": len(sources),
        "beam_size": beam_size,
        "bleu": metrics["bleu"],
        "chrf": metrics["chrf"],
        "seconds": elapsed,
        "sentences_per_second": len(sources) / max(1e-9, elapsed),
    }
    if save_hypotheses:
        save_hypotheses.parent.mkdir(parents=True, exist_ok=True)
        with open(save_hypotheses, "w", encoding="utf-8") as handle:
            for line in hypotheses:
                handle.write(line + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a translation checkpoint"
    )
    parser.add_argument("--model", required=True, help="checkpoint path")
    parser.add_argument(
        "--tokenizer", default=None, help="tokenizer.model path"
    )
    parser.add_argument(
        "--test-sets",
        default=",".join(DEFAULT_TEST_SETS),
        help="comma separated sacrebleu test set names",
    )
    parser.add_argument(
        "--directions",
        default="de-en,en-de",
        help="comma separated directions",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument(
        "--hypotheses-dir",
        default=None,
        help="directory for saving hypothesis files",
    )
    arguments = parser.parse_args(argv)

    translator = Translator.from_checkpoint(
        arguments.model, arguments.tokenizer, device=arguments.device
    )
    results = []
    for test_set in arguments.test_sets.split(","):
        for direction in arguments.directions.split(","):
            hypothesis_path = None
            if arguments.hypotheses_dir:
                hypothesis_path = (
                    Path(arguments.hypotheses_dir)
                    / f"{test_set}.{direction}.hyp.txt"
                )
            result = evaluate(
                translator,
                test_set.strip(),
                direction.strip(),
                beam_size=arguments.beam_size,
                batch_size=arguments.batch_size,
                max_sentences=arguments.max_sentences,
                save_hypotheses=hypothesis_path,
            )
            results.append(result)
            print(
                f"{result['test_set']} {result['direction']}: "
                f"BLEU {result['bleu']:.2f} | chrF {result['chrf']:.2f} | "
                f"{result['sentences_per_second']:.1f} sent/s"
            )

    payload = {
        "model": arguments.model,
        "parameters": translator.model.num_parameters(),
        "beam_size": arguments.beam_size,
        "results": results,
    }
    if arguments.output:
        output_path = Path(arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
