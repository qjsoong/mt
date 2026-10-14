# Training

This document describes the reference training run: data preparation, the
recipe, the hardware used and the measured cost. The goal is that anyone
with a similar consumer GPU can reproduce the released checkpoint.

## Hardware

| Resource | Value |
| -------- | ----- |
| GPU      | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, Ada, bfloat16) |
| CPU      | AMD Ryzen 9 7940H, 16 threads |
| RAM      | 30 GB |
| Storage  | ~20 GB free for corpora, tokenized data and checkpoints |

## Reproducing the run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[data,eval,dev]"

python scripts/download_data.py --corpora wmt14,opus100
python scripts/train_tokenizer.py --vocab-size 32000
python scripts/prepare_data.py
mt-train --config configs/mt-small-deen.yaml          # ~TBD h on the reference GPU
mt-eval --model artifacts/run/best.pt --test-sets wmt14,wmt19 \
        --output results/mt-small.json
```

Training is resumable:

```bash
mt-train --config configs/mt-small-deen.yaml --resume artifacts/run/last.pt
```

## Corpus statistics

Filled in from `data/processed/meta.json` of the reference run:

| Item | Value |
| ---- | ----- |
| Raw pairs read | TBD |
| Train pairs | TBD |
| Validation pairs | TBD |
| Train source tokens | TBD |
| Train target tokens | TBD |
| Rejected (length/quality/duplicate/tokenization) | TBD |

## Hyperparameters

| Item | Value |
| ---- | ----- |
| Parameters | ≈ 61 M |
| Batch size | 16 384 tokens per optimizer step (`max_tokens`), gradient accumulation 1 |
| Steps | TBD |
| Tokens seen | TBD |
| Epochs over the corpus | TBD |
| Learning rate | inverse square root, peak ≈ 7e-4 (`d_model^-0.5`), 4000 warm-up steps |
| Optimizer | AdamW, β=(0.9, 0.98), ε=1e-9, weight decay 0.01 |
| Label smoothing | 0.1 |
| Dropout / attention dropout | 0.1 / 0.1 |
| Gradient clipping | 1.0 |
| Precision | bfloat16 autocast |
| Validation | 64 batches of held-out pairs every 2000 steps |

## Measured cost

| Item | Value |
| ---- | ----- |
| Wall-clock training time | TBD |
| Throughput | TBD tokens/s (target tokens, averaged over the run) |
| Peak GPU memory | TBD |
| GPU energy estimate | TBD kWh (wall-clock × ~115 W board power) |
| CPU energy (data preparation) | negligible relative to training |

The energy estimate uses the nominal laptop GPU power limit; wall power and
therefore real consumption vary with the host system. It is an
order-of-magnitude figure, not a measurement.

## Practical notes

- The training script writes `artifacts/run/log.jsonl` (machine readable) and
  `config.yaml` (the exact resolved configuration).
- `best.pt` is selected by validation loss; `last.pt` supports resuming;
  `step_*.pt` checkpoints are rotated.
- On an 8 GB card, `max_tokens: 16384` leaves headroom for activations. If
  you see OOM, reduce `max_tokens` and increase `grad_accum_steps` to keep
  the effective batch size.
- For a quick pipeline check use `configs/smoke-test.yaml` (tiny model, 20
  steps).
