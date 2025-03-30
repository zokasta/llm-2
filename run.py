import torch
from transformers import BertTokenizer
from main import TinyLLM, generate_response

if __name__ == "__main__":
    
    tokenizer = BertTokenizer.from_pretrained("tiny_llm")  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model = TinyLLM(
        vocab_size=len(tokenizer),
        d_model=256,  
        seq_length=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
    )

    
    model.load_state_dict(
        torch.load("tiny_llm.pth", map_location=device)  
    )
    model.to(device)
    model.eval()

    
    print("Chat with the AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = generate_response(model, tokenizer, user_input, device)
        print("Bot:", response)
