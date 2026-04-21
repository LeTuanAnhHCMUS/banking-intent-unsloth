import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def main():
    print("Loading dataset...")
    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)

    # 1. Tập Train: Xáo trộn (shuffle) và lấy 10000 câu
    train_data = dataset["train"].shuffle(seed=42).select(range(10000))
    
    # 2. Tập Test: KHÔNG xáo trộn, lấy thẳng 1000 câu đầu tiên
    test_data = dataset["test"].select(range(1000))

    # Convert to pandas
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Label mapping (Chuyển ID nhãn thành chữ)
    label_names = dataset["train"].features["label"].names
    train_df["intent"] = train_df["label"].apply(lambda x: label_names[x])
    test_df["intent"] = test_df["label"].apply(lambda x: label_names[x])

    # Chỉ giữ lại text và intent
    train_df = train_df[["text", "intent"]]
    test_df = test_df[["text", "intent"]]

    # 3. Chia tập Train (10000) thành Train (9000) và Val (1000)
    # Dùng test_size=1000 để đảm bảo cắt chính xác số lượng
    train_df, val_df = train_test_split(
        train_df,
        test_size=1000, 
        stratify=train_df["intent"],
        random_state=42
    )

    # Đảm bảo thư mục lưu tồn tại
    os.makedirs("sample_data", exist_ok=True)

    # Lưu dữ liệu
    train_df.to_csv("sample_data/train.csv", index=False)
    val_df.to_csv("sample_data/val.csv", index=False)
    test_df.to_csv("sample_data/test.csv", index=False)

    print("✅ Done preprocessing!")
    print(f"👉 Số lượng Train thực tế: {len(train_df)} câu")
    print(f"👉 Số lượng Eval (val): {len(val_df)} câu")
    print(f"👉 Số lượng Test: {len(test_df)} câu")

if __name__ == "__main__":
    main()
