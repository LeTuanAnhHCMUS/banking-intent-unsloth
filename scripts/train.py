import yaml
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

def main():

    config = yaml.safe_load(open("configs/train.yaml"))

    train_df = pd.read_csv(config["train_file"])
    val_df = pd.read_csv(config["val_file"])

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
    )

    FastLanguageModel.for_training(model)

    def format_prompt(example):
        return {
            "text": f"""Classify intent

Message: {example['text']}

Intent: {example['intent']}
"""
        }

    train_dataset = train_dataset.map(format_prompt)
    val_dataset = val_dataset.map(format_prompt)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        args=TrainingArguments(
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            num_train_epochs=config["epochs"],
            warmup_steps=config["warmup_steps"],
            logging_steps=10,
            output_dir=config["output_dir"],
            evaluation_strategy="epoch"
        )
    )

    trainer.train()

    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

if __name__ == "__main__":
    main()
