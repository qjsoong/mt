# Data provenance

Training corpora are **not** committed to this repository. They are
downloaded by `scripts/download_data.py` and processed locally. This file
documents what is used, how it is filtered, and what license caveats apply.

## Sources

| Name      | Source                                                   | Pairs (raw) | Notes |
| --------- | -------------------------------------------------------- | ----------- | ----- |
| `wmt14`   | WMT 2014 German–English translation task, training split | ~4.5 M      | Europarl, News Commentary, Common Crawl |
| `opus100` | `Helsinki-NLP/opus-100`, `en-de` training split          | ~1.0 M      | OPUS aggregation of public corpora |

Both are distributed through the Hugging Face Hub:

- <https://huggingface.co/datasets/wmt/wmt14>
- <https://huggingface.co/datasets/Helsinki-NLP/opus-100>

WMT test sets (`wmt14`, `wmt19`, …) are fetched by `sacrebleu` at evaluation
time and are never used for training or model selection.

## Licensing

The WMT14 and OPUS corpora are collections of public data (parliamentary
proceedings, news commentary, web crawl text, subtitles, and so on). Each
underlying corpus has its own terms; the WMT/OPUS aggregations redistribute
them for research use. This repository only ships *code* and *derived
weights*; the raw corpora stay at their original distribution points.

Because the training data is a heterogeneous mixture, the released model
weights are published **for research use** and no warranty of fitness for a
particular purpose is given. Users who need a model for production use
should review the license of the constituent corpora, or retrain on their
own data with this pipeline.

## Filtering

`scripts/prepare_data.py` applies, in order:

1. Unicode normalization (NFC), HTML entity unescaping, control-character and
   zero-width-character removal, whitespace collapsing.
2. Character-length limits: at least 1 and at most 512 characters per side.
3. Length-ratio filter: `1/3 <= len(source)/len(target) <= 3`.
4. Letter-ratio filter: at least 50 % of the characters of each side must be
   alphabetic (drops markup and numeric noise, common in Common Crawl).
5. Exact-pair deduplication by CRC32 fingerprint.
6. SentencePiece tokenization; pairs longer than 128 subword tokens on either
   side are dropped.
7. A uniform hash-based validation split (`--dev-fraction`, 0.07 % by
   default).

The actual counts for the reference run (with rejection statistics) are
stored in `data/processed/meta.json` after preparation and are reported in
`docs/training.md`.

## What ships in the repository

| Artifact                  | Committed | Where |
| ------------------------- | --------- | ----- |
| Code, configs, docs       | yes       | git   |
| SentencePiece model       | yes (~0.5 MB) | `artifacts/tokenizer/tokenizer.model` (release asset / small enough for git) |
| Trained weights           | no        | GitHub release asset |
| Raw / processed corpora   | no        | `data/` (gitignored) |
| Training logs, checkpoints| no        | `artifacts/` (gitignored) |
| Evaluation results        | yes       | `results/*.json`, `docs/benchmarks.md` |
