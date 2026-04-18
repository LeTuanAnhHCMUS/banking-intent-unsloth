# banking-intent-unsloth

Fine-tuning project for intent classification on the **BANKING77** dataset using **Unsloth**.

## Project structure

```text
banking-intent-unsloth/
├── scripts/
│   ├── train.py
│   ├── inference.py
│   └── preprocess_data.py
├── configs/
│   ├── train.yaml
│   └── inference.yaml
├── sample_data/
│   ├── train.csv
│   └── test.csv
├── requirements.txt
├── train.sh
├── inference.sh
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> For Google Colab/Kaggle, run `pip install -r requirements.txt` in a notebook cell and then execute scripts with `!python ...`.

## Data preprocessing

Generate cleaned train/test CSV files from BANKING77:

```bash
python scripts/preprocess_data.py --output-dir sample_data --sample-size 5000
```

This script:
- Loads BANKING77 from HuggingFace datasets
- Cleans text
- Maps label IDs to label names
- Splits into train/test
- Saves CSV files to `sample_data/`

## Training

Update hyperparameters in `configs/train.yaml`, then run:

```bash
./train.sh
```

or:

```bash
python scripts/train.py --config configs/train.yaml
```

Training flow:
- Loads BANKING77 from HuggingFace datasets
- Samples a subset of data
- Preprocesses message text
- Fine-tunes with Unsloth + LoRA
- Saves checkpoint to `checkpoints/banking77_intent`

## Inference

Set checkpoint path in `configs/inference.yaml`, then run:

```bash
./inference.sh "my debit card payment is missing"
```

Python usage:

```python
from scripts.inference import IntentClassification

classifier = IntentClassification("checkpoints/banking77_intent")
print(classifier("where is my bank transfer"))
```

## Evaluation

A simple evaluation workflow:
1. Use `sample_data/test.csv` or regenerated test data from preprocessing.
2. Run model predictions for each test message.
3. Compare predicted intent vs `label_text` to compute metrics (accuracy/F1) with `scikit-learn`.

For quick experimentation in Colab/Kaggle, keep `subset_size` small in `configs/train.yaml` to reduce training time.
