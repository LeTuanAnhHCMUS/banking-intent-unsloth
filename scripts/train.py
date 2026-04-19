import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer


# ======================
# CONFIG
# ======================
MODEL_NAME = "unsloth/llama-3.2-3b-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 512

OUTPUT_DIR = "model"


# ======================
# PROMPT FORMAT (QUAN TRỌNG)
# ======================
def format_example(example):
    return {
        "text": f"""### Instruction:
Classify the banking intent.

### Input:
{example['text']}

### Response:
{example['intent']}"""
    }


def main():

    print("Loading dataset...")

    train_df = pd.read_csv("sample_data/train.csv")
    val_df = pd.read_csv("sample_data/val.csv")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    train_dataset = train_dataset.map(format_example)
    val_dataset = val_dataset.map(format_example)

    print("Loading model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # Enable LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing="unsloth",
    )

    print("Model ready for training...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,

            learning_rate=1e-4,

            logging_steps=10,
            save_steps=200,

            optim="adamw_8bit",
            weight_decay=0.01,

            lr_scheduler_type="linear",
            warmup_steps=50,

            fp16=True,
            report_to="none",  # tránh wandb popup
        ),
    )

    print("Starting training...")

    trainer.train()

    print("Saving model...")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done!")


if __name__ == "__main__":
    main()
