import argparse
import json
import re
from difflib import get_close_matches
from pathlib import Path

import torch
import yaml
from unsloth import FastLanguageModel


WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.strip().lower()
    return WHITESPACE_PATTERN.sub(" ", text)


def build_prompt(message: str) -> str:
    return (
        "Classify the banking intent for the customer message.\n"
        f"Message: {clean_text(message)}\n"
        "Intent:"
    )


class IntentClassification:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.model_path),
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        self.model.eval()

        mapping_file = self.model_path / "id2label.json"
        self.labels = []
        if mapping_file.exists():
            with mapping_file.open("r", encoding="utf-8") as handle:
                label_map = json.load(handle)
            self.labels = [label_map[key] for key in sorted(label_map, key=int)]

    def __call__(self, message):
        prompt = build_prompt(message)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = decoded.split("Intent:", maxsplit=1)[-1].strip().splitlines()[0].strip()

        if self.labels and predicted not in self.labels:
            matches = get_close_matches(predicted, self.labels, n=1, cutoff=0.0)
            if matches:
                predicted = matches[0]

        return predicted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run intent inference with a fine-tuned checkpoint")
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to inference YAML config")
    parser.add_argument("--message", required=True, help="Customer message to classify")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    model_path = config.get("model_path", "checkpoints/banking77_intent")
    classifier = IntentClassification(model_path=model_path)
    print(classifier(args.message))


if __name__ == "__main__":
    main()
