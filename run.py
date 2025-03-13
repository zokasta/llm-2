import torch
from transformers import BertTokenizer
from main import TinyTransformer

def generate_response(prompt):
    tokenizer = BertTokenizer.from_pretrained("mini_llm_model")
    vocab_size = tokenizer.vocab_size
    # Make sure seq_length matches what was used during training (128 here)
    model = TinyTransformer(vocab_size, d_model=128, seq_length=128, nhead=4)
    model.load_state_dict(torch.load("mini_llm_model/pytorch_model.bin", map_location=torch.device("cpu")))
    model.eval()
    tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = tokens["input_ids"]
    with torch.no_grad():
        output = model(input_ids)
    predicted_ids = output.argmax(dim=2)
    response_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return response_text

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print("Bot:", generate_response(user_input))
