import pandas as pd
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score
import torch

def main():

    model_name = "unsloth/llama-3.1-8b-unsloth-bnb-4bit"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        load_in_4bit=True
    )

    FastLanguageModel.for_inference(model)

    test_df = pd.read_csv("sample_data/test.csv")

    predictions = []
    labels = []

    for _, row in test_df.iterrows():

        prompt = f"""
Classify intent

Message: {row['text']}

Intent:
"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=10
        )

        pred = tokenizer.decode(outputs[0])

        predictions.append(pred)
        labels.append(row["intent"])

    acc = accuracy_score(labels, predictions)

    print("Base Model Accuracy:", acc)

if __name__ == "__main__":
    main()
