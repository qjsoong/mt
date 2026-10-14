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
mt-train --config configs/mt-small-deen.yaml          # 8.0 h on the reference GPU
mt-eval --model artifacts/run/best.pt --test-sets wmt14,wmt19 \
        --output results/mt-small.json
```

Training is resumable:

```bash
mt-train --config configs/mt-small-deen.yaml --resume artifacts/run/last.pt
```

## Corpus statistics

From `data/processed/meta.json` of the reference run:

| Item | Value |
| ---- | ----- |
| Raw pairs read | 5,508,785 (WMT14 4,508,785 + OPUS-100 1,000,000) |
| Train pairs | 5,160,716 |
| Validation pairs | 3,499 |
| Train source tokens | 142,650,538 |
| Train target tokens | 139,963,726 |
| Rejected | 344,570 (length 33,435; quality 200,148; duplicate 105,814; tokenization 5,173) |
| Kept by corpus | WMT14 4,315,133; OPUS-100 849,082 |

## Hyperparameters

| Item | Value |
| ---- | ----- |
| Parameters | 61,227,008 (≈ 61 M) |
| Batch size | 12,288 tokens per optimizer step (`max_tokens`), gradient accumulation 1 |
| Gradient checkpointing | on (needed to fit an 8 GB card) |
| Steps | 75,000 |
| Tokens seen | 921.6 M total, ≈ 456 M target tokens |
| Epochs over the corpus | ≈ 3.3 (each pair used once per pass, direction alternating) |
| Learning rate | inverse square root, peak ≈ 7e-4 (`d_model^-0.5`), 4000 warm-up steps |
| Optimizer | AdamW, β=(0.9, 0.98), ε=1e-9, weight decay 0.01 |
| Label smoothing | 0.1 |
| Dropout / attention dropout | 0.1 / 0.1 |
| Gradient clipping | 1.0 |
| Precision | bfloat16 autocast |
| Validation | 64 batches of held-out pairs every 2500 steps |
| Best validation loss | 3.276 (nats/token, label-smoothed) |

## Measured cost

| Item | Value |
| ---- | ----- |
| Wall-clock training time | 8.01 h (including validation runs) |
| Throughput | ≈ 14 k target tokens/s during training, ≈ 15.8 k target tokens/s counting all 921.6 M tokens over the full wall clock |
| Peak GPU memory | ≈ 5.0 GB with `max_tokens: 12288` and gradient checkpointing |
| GPU energy estimate | ≈ 0.9 kWh (8.0 h × ≈ 115 W nominal board power for an RTX 4060 Laptop) |
| CPU energy (data preparation) | negligible relative to training |

The energy estimate uses the nominal laptop GPU power limit; wall power and
therefore real consumption vary with the host system. It is an
order-of-magnitude figure, not a measurement. The training curve is in
`results/training-curve.png`.

## Practical notes

- The training script writes `artifacts/run/log.jsonl` (machine readable) and
  `config.yaml` (the exact resolved configuration).
- `best.pt` is selected by validation loss; `last.pt` supports resuming;
  `step_*.pt` checkpoints are rotated.
- On an 8 GB card, `max_tokens: 12288` with gradient checkpointing fits
  comfortably (~5 GB peak). Without checkpointing the same batch size needs
  about 6 GB and 16 384 tokens run out of memory. If you see OOM, reduce
  `max_tokens` and increase `grad_accum_steps` to keep the effective batch
  size.
- For a quick pipeline check use `configs/smoke-test.yaml` (tiny model, 20
  steps).
