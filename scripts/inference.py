import yaml
import torch
import re
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

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

class IntentClassification:
    def __init__(self, model_path):
        # model_path là đường dẫn tới file YAML config
        with open(model_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        checkpoint = config.get("model_path")
        if not checkpoint:
            raise ValueError("Config file must contain 'model_path' key pointing to your model checkpoint directory.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.params = config

    def __call__(self, message):
        # Format prompt PHẢI KHỚP VỚI LÚC TRAIN
        prompt = f"""### Instruction:\nClassify the banking intent.\n\n### Input:\n{message}\n\n### Response:\n"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                temperature=0.0
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Response:" in response:
            pred_raw = response.split("Response:")[-1].strip()
        else:
            pred_raw = response.strip()

        # Trích xuất nhãn chuẩn
        predicted_label = extract_intent(pred_raw, ALL_INTENTS)

        return predicted_label

# =========================
# SHORT USAGE EXAMPLE
# =========================
if __name__ == "__main__":
    # Suppress transformers warnings for clean output
    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()

    # Đảm bảo bạn đã tạo file configs/inference.yaml với nội dung:
    # model_path: "tên_thư_mục_chứa_model_của_bạn"

    config_file_path = "../configs/inference.yaml"

    print("Loading model for inference...")
    classifier = IntentClassification(model_path=config_file_path)

    # Cho phép người dùng nhập câu test_message từ bàn phím
    try:
        test_message = input("\n Enter intent message: ")
    except EOFError:
        test_message = "I lost my virtual card and I need a replacement."

    print("\n==============================")
    print(f"Input Message : {test_message}")
    predicted_intent = classifier(message=test_message)
    print(f"Predicted Intent: {predicted_intent}")
    print("==============================\n")