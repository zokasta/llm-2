import time
import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
from sacrebleu import corpus_bleu
import numpy as np

torch.backends.cudnn.benchmark = True

def load_dialogue_pairs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    user_buffer = []
    
    for entry in data:
        if entry["role"].lower() == "user":
            user_buffer.append(entry["prompt"].strip())
        elif entry["role"].lower() == "ai" and user_buffer:
            user_prompt = user_buffer.pop(0)
            ai_response = entry["prompt"].strip()
            pairs.append((user_prompt, ai_response))
    return pairs

class DialogueDataset(Dataset):
    def __init__(self, pairs, tokenizer, seq_length=128, augment=False):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _augment_text(self, text):
        if random.random() < 0.3:
            text = text.lower()
        if random.random() < 0.2:
            text = text.capitalize()
        return text

    def __getitem__(self, idx):
        user, assistant = self.pairs[idx]
        
        if self.augment:
            user = self._augment_text(user)
            assistant = self._augment_text(assistant)

        text = f"[USER]{user}[SEP][ASSISTANT]{assistant}[SEP]"
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.seq_length,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0)
        return input_ids, input_ids.clone()

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, seq_length=128, 
                 nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_length, d_model))
        self.dropout = nn.Dropout(dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.transformer(x)
        return self.fc(x)

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embedding
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
        new_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        
        old_num_tokens = old_embeddings.num_embeddings
        if old_num_tokens > 0:
            new_embeddings.weight.data[:old_num_tokens] = old_embeddings.weight.data[:old_num_tokens]
            
        self.embedding = new_embeddings
        return self

class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.AdamW(model.parameters(), lr=5e-4)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=5e-4, 
            steps_per_epoch=len(train_loader), epochs=20
        )
        self.scaler = torch.amp.GradScaler(device_type='cuda', enabled=device.type == "cuda")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', enabled=self.device.type == "cuda"):
                outputs = self.model(inputs)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1)
                )
                
            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1))
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

def generate_response(model, tokenizer, prompt, device, max_length=128):
    model.eval()
    prompt = f"[USER]{prompt}[SEP][ASSISTANT]"
    inputs = tokenizer(prompt, return_tensors="pt", 
                      max_length=max_length, truncation=True).to(device)
    
    generated = inputs.input_ids
    for _ in range(max_length):
        outputs = model(generated)
        next_token = outputs[:, -1, :].argmax(-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        if next_token == tokenizer.sep_token_id:
            break
    
    # Clean up the generated text
    full_response = tokenizer.decode(generated[0], skip_special_tokens=True)
    return full_response.split("[ASSISTANT]")[-1].replace("[SEP]", "").strip()

def evaluate_bleu(model, tokenizer, dataset, device, num_samples=100):
    references = []
    hypotheses = []
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in indices:
        user, true_resp = dataset.pairs[idx]
        generated = generate_response(model, tokenizer, user, device)
        
        # Handle empty responses
        generated = generated if len(generated) > 0 else " "
        true_resp = true_resp if len(true_resp) > 0 else " "
        
        references.append([true_resp])  # List of lists (multiple references per hypothesis)
        hypotheses.append(generated)     # List of strings

    return corpus_bleu(hypotheses, references).score

def main():
    SEED = 42
    SEQ_LENGTH = 128
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[USER]", "[ASSISTANT]", "[SEP]"]})
    
    pairs = load_dialogue_pairs("data/dialogue.json")
    random.shuffle(pairs)
    train_pairs, val_pairs = random_split(pairs, [0.9, 0.1])
    
    train_dataset = DialogueDataset(train_pairs, tokenizer, seq_length=SEQ_LENGTH, augment=True)
    val_dataset = DialogueDataset(val_pairs, tokenizer, seq_length=SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           num_workers=4, pin_memory=True)

    model = TinyLLM(
        vocab_size=len(tokenizer),
        d_model=256,
        seq_length=SEQ_LENGTH,
        nhead=8,
        num_layers=4,
        dropout=0.1
    )

    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    best_loss = float('inf')
    patience = 3
    no_improve = 0
    
    for epoch in range(20):
        start_time = time.time()
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:02} | Time: {epoch_time:.2f}s")
        print(f"\tTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print("Early stopping!")
            break

    model.load_state_dict(torch.load("best_model.pth"))
    bleu_score = evaluate_bleu(model, tokenizer, val_dataset, DEVICE)
    print(f"Validation BLEU Score: {bleu_score:.2f}")

    torch.save(model.state_dict(), "tiny_llm.pth")
    tokenizer.save_pretrained("tiny_llm")

if __name__ == "__main__":
    main()