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

def main():
    print("Loading dataset...")
    train_df = pd.read_csv("sample_data/train.csv")
    val_df = pd.read_csv("sample_data/val.csv")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # 1. ĐỊNH DẠNG PROMPT (Bắt buộc phải có eos_token)
    def format_example(example):
        return {
            "text": f"""### Instruction:
Classify the banking intent.

### Input:
{example['text']}

### Response:
{example['intent']}""" + tokenizer.eos_token
        }

    train_dataset = train_dataset.map(format_example)
    val_dataset = val_dataset.map(format_example)

    # 2. MỞ RỘNG NÃO BỘ CHO LORA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32, # Tăng alpha giúp mô hình tiếp thu nhanh hơn
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"], # Bổ sung MLP layers
        use_gradient_checkpointing="unsloth",
    )

    print("Model ready for training...")

    # 3. TỐI ƯU HYPERPARAMETERS
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False, # Tắt packing để tránh lẫn lộn context giữa các câu
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3, # Tăng epoch để học kỹ hơn 77 nhãn
            
            learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=100, # Rate chuẩn cho Llama
            
            logging_steps=10,
            save_steps=400,
            
            eval_strategy="steps", # Bật đánh giá validation
            eval_steps=200,
            
            optim="adamw_8bit",
            weight_decay=0.01,
            fp16=True,
            report_to="none",
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