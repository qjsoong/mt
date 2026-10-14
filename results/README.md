# Results

Evaluation results for the released checkpoint and the public baselines.
All numbers are produced by `scripts/run_benchmarks.sh`; the JSON files are
the raw outputs of those runs and `summary.md` is the generated table.

| File | Model | Test sets |
| ---- | ----- | --------- |
| `mt-small.json` | released `mt-small-deen` checkpoint | wmt14, wmt19 (both directions) |
| `baseline-opus-mt.de-en.json` | `Helsinki-NLP/opus-mt-de-en` | wmt14, wmt19 |
| `baseline-opus-mt.en-de.json` | `Helsinki-NLP/opus-mt-en-de` | wmt14, wmt19 |

Each file records the model, parameter count, beam size, and for every test
set: sentence count, BLEU, chrF, wall-clock seconds and sentences per
second. Regenerate everything with:

```bash
scripts/run_benchmarks.sh artifacts/release/mt-small-deen.pt
```

The tables rendered into `docs/benchmarks.md` are copied from
`summary.md`.
