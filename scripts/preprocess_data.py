from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():

    print("Loading dataset...")

    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)

    train_data = dataset["train"]
    test_data = dataset["test"]

    # Sample subset
    train_data = train_data.shuffle(seed=42).select(range(4000))
    test_data = test_data.shuffle(seed=42).select(range(800))

    # Convert to pandas
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # label mapping
    label_names = dataset["train"].features["label"].names

    train_df["intent"] = train_df["label"].apply(lambda x: label_names[x])
    test_df["intent"] = test_df["label"].apply(lambda x: label_names[x])

    train_df = train_df[["text", "intent"]]
    test_df = test_df[["text", "intent"]]

    # validation split
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["intent"],
        random_state=42
    )

    os.makedirs("sample_data", exist_ok=True)

    train_df.to_csv("sample_data/train.csv", index=False)
    val_df.to_csv("sample_data/val.csv", index=False)
    test_df.to_csv("sample_data/test.csv", index=False)

    print("Done preprocessing")


if __name__ == "__main__":
    main()
