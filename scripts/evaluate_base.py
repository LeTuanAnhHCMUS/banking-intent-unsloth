import pandas as pd
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score
import torch


INTENT_LIST = """
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

def main():

    print("Loading model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/llama-3.2-3b-unsloth-bnb-4bit",
        max_seq_length=512,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    print("Loading test data...")
    test_df = pd.read_csv("sample_data/test.csv")

    predictions = []
    labels = []

    print("Evaluating base model...")

    for _, row in test_df.iterrows():

        prompt = f"""
You are a banking intent classifier.

Choose ONLY one intent from the list below.

{INTENT_LIST}

Text: {row['text']}

Intent:
"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("MODEL OUTPUT:")
        print(response)
        print("="*50)
        pred = response.split("Intent:")[-1].strip().split("\n")[0]

        # Clean prediction
        pred = pred.replace(".", "").strip()

        predictions.append(pred)
        labels.append(row["intent"])

    acc = accuracy_score(labels, predictions)

    print("\nBase Model Accuracy:", acc)

    # Print sample predictions
    print("\nSample predictions:\n")

    for i in range(5):
        print("Text:", test_df.iloc[i]["text"])
        print("True:", labels[i])
        print("Pred:", predictions[i])
        print("-" * 50)


if __name__ == "__main__":
    main()
