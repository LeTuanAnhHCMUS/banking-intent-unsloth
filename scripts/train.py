import argparse
import json
import re
from pathlib import Path

import yaml
from datasets import concatenate_datasets, load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


WHITESPACE_PATTERN = re.compile(r"\s+")


DEFAULT_CONFIG = {
    "dataset_name": "PolyAI/banking77",
    "model_name": "unsloth/phi-2-bnb-4bit",
    "output_dir": "checkpoints/banking77_intent",
    "subset_size": 1500,
    "test_size": 0.1,
    "random_seed": 42,
    "max_seq_length": 1024,
    "load_in_4bit": True,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 1,
    "learning_rate": 2e-4,
    "warmup_steps": 10,
    "logging_steps": 10,
    "save_steps": 50,
    "eval_steps": 50,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
}


def clean_text(text: str) -> str:
    text = text.strip().lower()
    return WHITESPACE_PATTERN.sub(" ", text)


def build_prompt(message: str, label: str) -> str:
    return (
        "Classify the banking intent for the customer message.\n"
        f"Message: {message}\n"
        f"Intent: {label}"
    )


def load_config(path: str) -> dict:
    config = DEFAULT_CONFIG.copy()
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config.update(loaded)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an intent model on BANKING77 with Unsloth")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config["dataset_name"])
    label_names = dataset["train"].features["label"].names

    merged = concatenate_datasets([dataset["train"], dataset["test"]])
    merged = merged.shuffle(seed=config["random_seed"])
    subset_size = int(config["subset_size"])
    if subset_size and subset_size < len(merged):
        merged = merged.select(range(subset_size))

    def formatter(example):
        message = clean_text(example["text"])
        label = label_names[int(example["label"])]
        return {"text": build_prompt(message, label)}

    formatted = merged.map(formatter, remove_columns=merged.column_names)
    split_dataset = formatted.train_test_split(
        test_size=config["test_size"],
        seed=config["random_seed"],
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=bool(config["load_in_4bit"]),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(config["lora_r"]),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_output"),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        warmup_steps=int(config["warmup_steps"]),
        num_train_epochs=float(config["num_train_epochs"]),
        logging_steps=int(config["logging_steps"]),
        save_steps=int(config["save_steps"]),
        eval_steps=int(config["eval_steps"]),
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to="none",
        seed=int(config["random_seed"]),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        dataset_text_field="text",
        max_seq_length=int(config["max_seq_length"]),
        packing=False,
        args=training_args,
    )

    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with (output_dir / "id2label.json").open("w", encoding="utf-8") as handle:
        json.dump({str(i): label for i, label in enumerate(label_names)}, handle, indent=2, ensure_ascii=False)

    print(f"Model checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    main()
