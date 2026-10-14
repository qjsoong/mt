# Contributing

Thanks for your interest in `mt`. This is a small, focused project: a
from-scratch German↔English translation model with a reproducible training
pipeline. Contributions that keep it that way are welcome.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[data,eval,dev]"
# GPU users: install the CUDA build of PyTorch first from pytorch.org
```

## Checks

Run the same checks as CI before opening a pull request:

```bash
ruff format --check src tests scripts
ruff check src tests scripts
pytest
```

`pytest` runs on CPU and takes a couple of minutes (it trains a tiny
SentencePiece model and a tiny Transformer).

## Guidelines

- Keep the package dependency-light: core training/inference only needs
  `torch`, `numpy`, `sentencepiece`, `PyYAML` and `tqdm`. Put data
  preparation and evaluation dependencies behind the `data` and `eval`
  extras.
- Add a test for behaviour changes in the model, mask handling, batching or
  decoding. Model changes should keep `test_incremental_decoding_matches_full_forward`
  passing.
- Document user-visible changes in the README or `docs/`.
- Do not commit corpora, checkpoints, logs or anything derived from
  licensed data; see `docs/data.md` for what ships in the repository.

## Reporting results

If you train a model with this pipeline, the `results/` directory shows the
expected JSON shape. Benchmarks should use `sacrebleu` test sets and the
decoding settings in `docs/benchmarks.md` so numbers stay comparable.
