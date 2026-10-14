#!/usr/bin/env python3
"""Train the shared German/English SentencePiece vocabulary.

Reads the raw JSONL corpora produced by ``download_data.py`` and writes a
single text file per language, then trains a unigram SentencePiece model
with the special tokens required by the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mt.tokenizer import train_tokenizer  # noqa: E402


def extract_texts(raw_dir: Path, data_dir: Path) -> tuple[Path, Path]:
    german_path = data_dir / "tokenizer_corpus.de.txt"
    english_path = data_dir / "tokenizer_corpus.en.txt"
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no .jsonl files found in {raw_dir}")
    print(f"extracting text from {len(files)} file(s)")
    count = 0
    with (
        open(german_path, "w", encoding="utf-8") as german_handle,
        open(english_path, "w", encoding="utf-8") as english_handle,
    ):
        for path in files:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    german_handle.write(record["source"].strip() + "\n")
                    english_handle.write(record["target"].strip() + "\n")
                    count += 1
                    if count % 500000 == 0:
                        print(f"  {count:,} pairs", flush=True)
    print(f"wrote {count:,} sentence pairs per language")
    return german_path, english_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="artifacts/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=8000000,
        help="sample this many sentences from the combined text files",
    )
    parser.add_argument("--num-threads", type=int, default=16)
    arguments = parser.parse_args()

    german_path, english_path = extract_texts(
        Path(arguments.raw_dir), Path(arguments.data_dir)
    )
    out_dir = Path(arguments.out_dir)
    model_path = train_tokenizer(
        [german_path, english_path],
        out_dir / "tokenizer",
        vocab_size=arguments.vocab_size,
        num_threads=arguments.num_threads,
        max_sentences=arguments.max_sentences,
    )
    from mt.tokenizer import MTTokenizer

    tokenizer = MTTokenizer(model_path)
    tokenizer.save_meta(out_dir / "meta.json")
    print(
        f"tokenizer written to {model_path} "
        f"(vocab_size={tokenizer.vocab_size}, "
        f"lang tokens={tokenizer.lang_token_ids})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
