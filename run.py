import torch
from transformers import BertTokenizer
from main import TinyTransformer, generate_response


def generate_response(prompt, model, device, seq_length=128):
    tokens = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=seq_length,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    with torch.no_grad():
        output = model(input_ids)
    predicted_ids = output.argmax(dim=2)
    response = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("mini_llm_model")
    vocab_size = tokenizer.vocab_size

    model = TinyTransformer(vocab_size, d_model=128, seq_length=128, nhead=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load("mini_llm_model/pytorch_model.bin", map_location=device)
    )
    model.to(device)
    model.eval()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        prompt = f"User: {user_input}\nAssistant:"
        with torch.no_grad():
            tokens = tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(device)
            output = model(input_ids)
            predicted_ids = output.argmax(dim=2)
            response = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        print("Bot:", response)
