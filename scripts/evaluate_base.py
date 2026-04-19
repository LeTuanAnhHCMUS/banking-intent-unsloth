import pandas as pd
import re
import torch
from sklearn.metrics import accuracy_score
from unsloth import FastLanguageModel

# =========================
# CONFIG & LABELS
# =========================
# Trỏ trực tiếp đến model gốc trên HuggingFace
MODEL_PATH = "unsloth/llama-3.2-3b-unsloth-bnb-4bit" 
TEST_PATH = "sample_data/test.csv"

# Danh sách 77 intent cố định
RAW_INTENT_LIST = """
activate_my_card
age_limit
apple_pay_or_google_pay
atm_support
automatic_top_up
balance_not_updated_after_bank_transfer
balance_not_updated_after_cheque_or_cash_deposit
beneficiary_not_allowed
cancel_transfer
card_about_to_expire
card_acceptance
card_arrival
card_delivery_estimate
card_linking
card_not_working
card_payment_fee_charged
card_payment_not_recognised
card_payment_wrong_exchange_rate
card_swallowed
cash_withdrawal_charge
cash_withdrawal_not_recognised
change_pin
compromised_card
contactless_not_working
country_support
declined_card_payment
declined_cash_withdrawal
declined_transfer
direct_debit_payment_not_recognised
disposable_card_limits
edit_personal_details
exchange_charge
exchange_rate
exchange_via_app
extra_charge_on_statement
failed_transfer
fiat_currency_support
get_disposable_virtual_card
get_physical_card
getting_spare_card
getting_virtual_card
lost_or_stolen_card
lost_or_stolen_phone
order_physical_card
passcode_forgotten
pending_card_payment
pending_cash_withdrawal
pending_top_up
pending_transfer
pin_blocked
receiving_money
Refund_not_showing_up
request_refund
reverted_card_payment
supported_cards_and_currencies
terminate_account
top_up_by_bank_transfer_charge
top_up_by_card_charge
top_up_by_cash_or_cheque
top_up_failed
top_up_limits
top_up_reverted
topping_up_by_card
transaction_charged_twice
transfer_fee_charged
transfer_into_account
transfer_not_received_by_recipient
transfer_timing
unable_to_verify_identity
verify_my_identity
verify_source_of_funds
verify_top_up
virtual_card_not_working
visa_or_mastercard
why_verify_identity
wrong_amount_of_cash_received
wrong_exchange_rate_for_cash_withdrawal
"""
ALL_INTENTS = [label.strip().lower() for label in RAW_INTENT_LIST.strip().split("\n") if label.strip()]

# Tạo chuỗi danh sách nhãn để chèn vào prompt cho Base Model
LABELS_STRING_FOR_PROMPT = "\n".join([f"- {label}" for label in ALL_INTENTS])

# =========================
# HELPERS
# =========================
def normalize(text):
    return str(text).lower().strip().replace(" ", "_").replace(".", "")

def extract_intent(pred_text, label_list):
    pred_text = pred_text.lower().strip()
    
    for label in label_list:
        if label in pred_text:
            return label
            
    tokens = re.findall(r"[a-z_]+", pred_text)
    for t in tokens:
        if t in label_list:
            return t
            
    candidates = [t for t in tokens if "_" in t]
    if candidates:
        return max(candidates, key=len)
        
    return pred_text

# PROMPT CHO BASE MODEL (Zero-shot)
def build_base_prompt(text):
    return f"""You are a strict banking intent classifier.

Return EXACTLY ONE label from the list below.
Do NOT add explanation or punctuation.

Labels:
{LABELS_STRING_FOR_PROMPT}

Text:
{text}

Answer:
"""

# =========================
# MAIN EXECUTION
# =========================
def main():
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=512,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)

    predictions = []
    labels = []

    print("\nEvaluating base model...\n")

    for i, row in test_df.iterrows():
        prompt = build_base_prompt(row["text"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                temperature=0.0
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Lấy phần text sau "Answer:"
        if "Answer:" in response:
            pred_raw = response.split("Answer:")[-1].strip()
        else:
            pred_raw = response.strip()
        
        # Lọc nhãn an toàn
        pred = extract_intent(pred_raw, ALL_INTENTS)
        
        pred = normalize(pred)
        true = normalize(row["intent"])

        predictions.append(pred)
        labels.append(true)

        correct = "✓" if pred == true else "✗"
        print(f"Sample {i+1} [{correct}]")
        print("Text :", row["text"])
        print("True :", true)
        print("Pred :", pred)
        print("-" * 60)

    acc = accuracy_score(labels, predictions)
    print("\n============================")
    print("Base Model Accuracy:", acc)
    print("============================\n")

if __name__ == "__main__":
    main()