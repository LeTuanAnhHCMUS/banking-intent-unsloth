import argparse
import json
import re
from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split


WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.strip().lower()
    return WHITESPACE_PATTERN.sub(" ", text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BANKING77 dataset as CSV files.")
    parser.add_argument("--dataset-name", default="PolyAI/banking77", help="Hugging Face dataset name")
    parser.add_argument("--output-dir", default="sample_data", help="Directory to store train/test CSV files")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional number of samples to keep")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_name)
    merged = concatenate_datasets([dataset["train"], dataset["test"]])

    if args.sample_size and args.sample_size < len(merged):
        merged = merged.shuffle(seed=args.random_state).select(range(args.sample_size))

    label_names = dataset["train"].features["label"].names

    rows = []
    for item in merged:
        rows.append(
            {
                "text": clean_text(item["text"]),
                "label": int(item["label"]),
                "label_text": label_names[int(item["label"])],
            }
        )

    frame = pd.DataFrame(rows)
    label_counts = frame["label"].value_counts()
    stratify_labels = frame["label"] if (not label_counts.empty and label_counts.min() >= 2) else None
    train_df, test_df = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_labels,
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    label_map = {str(index): label for index, label in enumerate(label_names)}
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2, ensure_ascii=False)

    print(f"Saved {len(train_df)} train samples to {output_dir / 'train.csv'}")
    print(f"Saved {len(test_df)} test samples to {output_dir / 'test.csv'}")


if __name__ == "__main__":
    main()
