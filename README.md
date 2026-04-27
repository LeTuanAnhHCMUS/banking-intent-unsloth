# Banking Intent Classification with Llama 3 & Unsloth
Link Drive folder: https://drive.google.com/drive/folders/1eekuX65RQze23xVkZIe-P_1i16_euzhg?usp=sharing  
Link Kaggle notebook: https://www.kaggle.com/code/ltanh64/lab2nlpinindustry  
Link download model: https://huggingface.co/Tuan-Anh-64/banking-intent-llama3  

Fine-tune **Llama 3** (QLoRA) cho bài toán phân loại 77 banking intents trên tập BANKING77.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Training Configuration](#training-configuration)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Dataset](#dataset)
- [Troubleshooting](#troubleshooting)

---

## Project Structure

```
banking-intent-unsloth/
├── scripts/
│   ├── train.py                # Fine-tune script (Unsloth QLoRA, Trainable parameters = 24,313,856 of 3,237,063,680 (0.75% trained))
│   ├── inference.py            # Unified inference (class + CLI)
|   ├── evaluate_base.py        # Accuracy evaluation based model
│   ├── evaluate_finetuned.py   # Accuracy evaluation finetuned model
│   └── preprocess_data.py      # Download & preprocess Banking77
├── configs/
│   ├── train.yaml              # Training config (model, LoRA, hyperparams)
│   └── inference.yaml          # Inference config (generation, paths)
├── sample_data/                # Dữ liệu sau khi chạy preprocess
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── outputs/
│   └── banking_intent_model/   # Model sau khi train
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Installation

```bash
# 1. Clone repository
!git clone <repo_url>
%cd banking-intent-unsloth

# 2. Install dependencies
!pip install -r requirements.txt
```

## Training Configuration

### Model Configuration (`configs/train.yaml`)

```yaml
model:
  name: "unsloth/llama-3.2-3b-unsloth-bnb-4bit"
  max_seq_length: 512
  load_in_4bit: true
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  learning_rate: 2e-4
  lr_scheduler_type: "cosine"
  warmup_steps: 100
  logging_steps: 10
  save_steps: 400
  eval_strategy: "steps"
  eval_steps: 200
  optim: "adamw_8bit"
  weight_decay: 0.01
  fp16: true
  report_to: "none"
output_dir: "outputs/banking_intent_model"
```

---

## Training

```bash
# 1. Preprocess data
python scripts/preprocess_data.py

# 2. Train model
python scripts/train.py
```

- Tập dữ liệu sẽ được tải về, làm sạch và lưu vào sample_data/.
- Model checkpoint sẽ được lưu vào outputs/banking_intent_model/.

---

## Inference

### Python API

```python
from scripts.inference import IntentClassification

classifier = IntentClassification("configs/inference.yaml")
label = classifier("I lost my virtual card and I need a replacement.")
print(label)  # → transfer_into_account
```

### CLI

```bash
python scripts/inference.py
```

---

## Evaluation

```bash
python scripts/evaluate_finetuned.py
```

- Script sẽ chạy dự đoán trên sample_data/test.csv và in ra độ chính xác.

---

## Dataset

- **Banking77** từ `PolyAI/banking77` trên HuggingFace
- 77 intent classes
- ~13000 samples (10003 train samples and 3080 test samples)
- Tự động download khi chạy preprocess_data.py

---

## Troubleshooting

- Nếu lỗi thiếu thư viện: `pip install -r requirements.txt`
- Nếu lỗi CUDA: kiểm tra GPU, RAM, CUDA version
- Nếu lỗi FileNotFound: kiểm tra lại đường dẫn model, data, config
- Nếu lỗi encoding YAML: lưu file yaml với encoding UTF-8 hoặc đọc file yaml với encoding UTF-8 

---

## Requirements

```
unsloth
transformers
datasets
accelerate
peft
trl
torch
pandas
scikit-learn
pyyaml
```
