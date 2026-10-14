# mt — a small machine translation model, built from scratch

`mt` is a compact German↔English neural machine translation model designed,
trained and evaluated in the open. The model architecture, tokenizer, data
pipeline and beam-search decoder are implemented from scratch with PyTorch —
no pre-trained weights are used at any point.

The whole project is designed to be reproducible on a single consumer GPU
(8 GB VRAM). Training data comes from public corpora (WMT14 and OPUS-100);
the trained checkpoint is published as a release asset rather than committed
to the repository.

## Results

See [docs/benchmarks.md](docs/benchmarks.md) for the full comparison with
public models of comparable size (`Helsinki-NLP/opus-mt-*`), including BLEU,
chrF and throughput numbers on standard WMT test sets.

| Test set | Direction | mt-small | opus-mt |
| -------- | --------- | -------- | ------- |
| wmt14    | de→en     | TBD      | TBD     |
| wmt14    | en→de     | TBD      | TBD     |
| wmt19    | de→en     | TBD      | TBD     |
| wmt19    | en→de     | TBD      | TBD     |

## Install

```bash
git clone https://github.com/<user>/mt.git
cd mt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[data,eval]"      # installs PyTorch, sentencepiece, ...
# for CUDA wheels, install torch first from https://pytorch.org
```

## Translate with the released model

```bash
# download the release assets (checkpoint + tokenizer)
scripts/download_model.sh

mt-translate \
    --model artifacts/release/model.pt \
    --tokenizer artifacts/release/tokenizer.model \
    --direction de-en \
    --input examples/de.txt
```

Python API:

```python
from mt.generate import Translator

translator = Translator.from_checkpoint(
    "artifacts/release/model.pt", "artifacts/release/tokenizer.model"
)
print(translator.translate(["Der Hund schläft."], direction="de-en"))
```

## Train from scratch

```bash
# 1. Python environment (see pyproject.toml for extras)
pip install -e ".[data,eval,dev]"

# 2. Raw corpora -> data/raw/*.jsonl
python scripts/download_data.py --corpora wmt14,opus100

# 3. Shared SentencePiece vocabulary -> artifacts/tokenizer/
python scripts/train_tokenizer.py --vocab-size 32000

# 4. Tokenized, memory-mapped training data -> data/processed/
python scripts/prepare_data.py

# 5. Train
mt-train --config configs/mt-small-deen.yaml

# 6. Evaluate
mt-eval --model artifacts/run/best.pt --test-sets wmt14,wmt19
```

See [docs/training.md](docs/training.md) for details on the recipe, hardware
usage and expected runtimes, and [docs/design.md](docs/design.md) for the
architecture and batching design.

## Repository layout

```
configs/        training configurations
docs/           design, training, data and benchmark documentation
scripts/        data download, tokenizer training, baselines
src/mt/         the Python package (model, data, train, generate, evaluate)
tests/          unit tests for model, data and decoding
results/        evaluation results for the released checkpoint and baselines
```

## Development

```bash
pip install -e ".[dev]"
ruff format src tests scripts
ruff check src tests scripts
pytest
```

## License and data

Code is released under the Apache-2.0 license. Training corpora are
downloaded at runtime and are not redistributed by this repository; see
[docs/data.md](docs/data.md) for provenance and licensing notes. Model
weights are published for research use with the same data provenance
caveats.
