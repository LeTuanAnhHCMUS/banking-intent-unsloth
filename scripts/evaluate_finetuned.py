import pandas as pd
import torch
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score


def normalize(text):
    return str(text).lower().strip().replace(" ", "_").replace(".", "")


def build_prompt(text):
    return f"""### Instruction:
Classify the banking intent from the user message.

### Input:
{text}

### Response:
"""


def main():
    print("Loading fine-tuned model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="model",   # thư mục output sau khi train
        max_seq_length=512,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    print("Loading test data...")
    test_df = pd.read_csv("sample_data/test.csv")

    predictions = []
    labels = []

    print("\nEvaluating fine-tuned model...\n")

    for i, row in test_df.iterrows():

        prompt = build_prompt(row["text"])

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # lấy phần sau "Response:"
        pred = response.split("Response:")[-1].strip().split("\n")[0]
        pred = normalize(pred)
        true = normalize(row["intent"])

        predictions.append(pred)
        labels.append(true)

        # 🔥 PRINT từng sample để debug
        correct = "✓" if pred == true else "✗"

        print(f"Sample {i+1} [{correct}]")
        print("Text :", row["text"])
        print("True :", true)
        print("Pred :", pred)
        print("-" * 60)

    acc = accuracy_score(labels, predictions)

    print("\n============================")
    print("Fine-tuned Accuracy:", acc)
    print("============================\n")

    # show vài ví dụ cuối
    print("Sample predictions (last 5):\n")

    for i in range(-5, 0):
        print("TEXT:", test_df.iloc[i]["text"])
        print("TRUE:", labels[i])
        print("PRED:", predictions[i])
        print("-" * 50)


if __name__ == "__main__":
    main()
