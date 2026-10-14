# Benchmarks

This document compares the released `mt-small-deen` checkpoint against public
translation models of comparable size on standard test sets. All numbers are
produced by the scripts in this repository and the raw JSON is stored in
`results/`.

## Methodology

- **Test sets.** `wmt14` (newstest2014) and `wmt19` (newstest2019), fetched
  with `sacrebleu`, which also applies the standard tokenization for BLEU
  (13a) and chrF2. Neither test set is used for training or model selection;
  the dev split used during training is held out from the training corpus.
- **Directions.** `de→en` and `en→de`, evaluated separately. The `mt` model
  is a single checkpoint for both directions.
- **Decoding.** Beam size 5, GNMT length penalty `alpha = 0.6`, batch size
  64, maximum output length `1.5 × source tokens + 20`. Baselines use the
  same beam size and batch size through `transformers.generate`.
- **Metrics.** `sacrebleu` BLEU and chrF2, plus sentences per second measured
  on the same machine (RTX 4060 Laptop, 8 GB).
- **Parameters** are counted as unique trainable weights (tied embeddings
  counted once).

## Models

| Model | Parameters | Training data (public description) |
| ----- | ---------- | ---------------------------------- |
| `mt-small-deen` (ours) | ≈ 61 M | WMT14 de-en train + OPUS-100 en-de train |
| `Helsinki-NLP/opus-mt-de-en` | 74.4 M | OPUS corpora |
| `Helsinki-NLP/opus-mt-en-de` | 74.4 M | OPUS corpora |

The comparison is deliberately *not* compute-matched: OPUS models are
trained on much larger, continuously updated corpora. The `mt` checkpoint is
produced on a single 8 GB consumer GPU in a fraction of the time. The tables
show what is achievable under that budget.

## Results

All numbers were produced with beam size 5 on the same machine (RTX 4060
Laptop, 8 GB). `sent/s` is sentences per second on the corresponding test
set; the `mt` numbers include tokenization and detokenization.

### de→en

| Model | Params | wmt14 BLEU | wmt14 chrF | wmt19 BLEU | wmt19 chrF | sent/s |
| --- | --- | --- | --- | --- | --- | --- |
| mt-small-deen | 61.2M | 26.31 | 53.74 | 30.69 | 56.11 | 34.4 |
| Helsinki-NLP/opus-mt-de-en | 74.4M | 33.99 | 59.98 | 39.45 | 63.89 | 15.9 |

### en→de

| Model | Params | wmt14 BLEU | wmt14 chrF | wmt19 BLEU | wmt19 chrF | sent/s |
| --- | --- | --- | --- | --- | --- | --- |
| mt-small-deen | 61.2M | 21.20 | 52.30 | 30.83 | 57.06 | 34.8 |
| Helsinki-NLP/opus-mt-en-de | 74.4M | 27.58 | 58.39 | 40.51 | 65.17 | 26.0 |

Raw JSON: `results/mt-small.json`, `results/baseline-opus-mt.de-en.json`,
`results/baseline-opus-mt.en-de.json`.

## How to read these numbers

- `mt-small-deen` reaches **~77 % of the opus-mt BLEU score** on every test
  set and direction, with 82 % of the parameters, while being trained from
  scratch on ~5.2 M sentence pairs in **8 GPU-hours on a single laptop
  GPU**. The opus-mt models are trained on much larger, continuously
  maintained OPUS corpora and have more parameters; the comparison shows
  what is achievable in this budget, not parity.
- The gap is consistent across test sets (76–78 % relative BLEU), which
  suggests the model is not overfitting to one domain.
- Both models do better on wmt19 than on wmt14; newstest2014 is the harder
  set for this data mixture.
- The custom KV-cached beam search is faster than the Hugging Face
  baseline at the same beam size on this hardware, despite a smaller batch
  size for the baseline (32 vs 64).
- Table regeneration: `scripts/run_benchmarks.sh`.

