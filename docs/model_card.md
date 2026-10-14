# Model card: mt-small-deen

## Model details

- **Name:** mt-small-deen
- **Version:** v0.1.0
- **Type:** encoder–decoder Transformer, trained from scratch
- **Languages:** German ↔ English, one bilingual checkpoint
- **Parameters:** ≈ 61 M
- **Architecture:** 6 encoder + 6 decoder layers, `d_model = 512`, 8 heads,
  SwiGLU feed-forward (hidden 1408), RMSNorm, sinusoidal positions, shared
  SentencePiece vocabulary (32k, unigram), tied input/output embeddings
- **Training data:** WMT14 de-en training split + OPUS-100 en-de training
  split (public corpora, see `docs/data.md`)
- **Training hardware:** a single 8 GB consumer GPU (RTX 4060 Laptop)
- **License:** Apache-2.0 for the code; weights released for research use,
  subject to the licenses of the underlying corpora
- **Repository:** the GitHub repository hosting this file

## Intended use

- Research on small, reproducible machine translation models.
- Translation of short and medium-length sentences between German and
  English, both directions.
- As a baseline or starting point for fine-tuning on domain-specific data.

## Out-of-scope use

- Languages other than German and English.
- Safety-critical, medical, legal or otherwise high-stakes translation
  without human review.
- Long documents without sentence splitting (the model is trained on
  sentence pairs up to 128 subword tokens).
- Formality control, terminology enforcement, or any constrained decoding.

## Training

| Item                | Value |
| ------------------- | ----- |
| Tokenizer           | 32k SentencePiece unigram, byte fallback |
| Train pairs         | see `docs/training.md` |
| Directions          | de→en and en→de, one target-language token |
| Max tokens / batch  | 16384 |
| Optimizer           | AdamW, β=(0.9, 0.98), ε=1e-9 |
| LR schedule         | inverse square root, 4000 warm-up steps |
| Label smoothing     | 0.1 |
| Dropout             | 0.1 |
| Precision           | bfloat16 autocast, FP32 master weights |
| Steps / tokens seen | see `docs/training.md` |

Model selection uses a held-out split of the training corpus, never the WMT
test sets.

## Evaluation

Evaluation uses `sacrebleu` test sets with the same decoding settings for
this model and the baselines. Results are in `docs/benchmarks.md` and the
raw JSON files in `results/`.

## Limitations

- Trained on a few million sentence pairs; rare words, named entities and
  long-range dependencies are less reliable than in larger systems.
- The training mixture contains web-crawled data, so some noise and bias
  from the underlying corpora is inherited.
- Language direction is selected by an explicit token; passing the wrong
  direction produces poor translations rather than an error.
- No document-level context: each sentence is translated independently.

## Environmental impact

The reference run was trained on a single 8 GB laptop GPU. Estimated energy
consumption and carbon footprint are reported in `docs/training.md` based on
measured wall-clock time and the nominal power draw of the device.
