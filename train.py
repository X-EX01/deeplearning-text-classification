import torch
import os
import time
import json
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from datasets import load_dataset

from transformer import get_transformer_components, prepare_transformer_data
from rnn import build_vocab_and_embeddings, get_rnn_dataloader, BiLSTM, BiGRU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, model_name, is_transformer=True, epochs=2):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5 if is_transformer else 1e-3)
    loss_fn = CrossEntropyLoss()

    start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        model.train()
        print(f"\n--- {model_name} | Epoch {epoch+1}/{epochs} ---")
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            if is_transformer:
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            else:
                logits = model(input_ids)
                
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})
            
    train_time = time.perf_counter() - start_time
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    
    return model, train_time, max_memory

def main():
    print(f"Using device: {device}")
    metrics = {}

    print("Loading DBpedia14 and extracting training subset...")
    raw_dataset = load_dataset("dbpedia_14")
    
    # Using 28,000 => exactly 2,000 per class
    train_data = raw_dataset['train'].train_test_split(train_size=28000, stratify_by_column="label")['train']

    # --- Train Transformers ---
    transformer_configs = [
        ("DistilBERT", "distilbert-base-uncased"),
        ("RoBERTa", "distilroberta-base")
    ]

    for name, model_id in transformer_configs:
        print(f"\n=== TRANSFORMER ({name}) ===")
        tokenizer, model = get_transformer_components(model_id)
        
        train_loader = prepare_transformer_data(train_data, tokenizer, shuffle=True)
        
        metrics[f'{name}_Params'] = count_parameters(model)
        model, t_time, t_mem = train_model(model, train_loader, name, is_transformer=True, epochs=2)
        metrics[f'{name}_Train_Time_s'] = t_time
        metrics[f'{name}_Train_Mem_MB'] = t_mem
        
        save_path = f"models/{name.lower()}"
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        del model, train_loader; torch.cuda.empty_cache()

    print("\n=== PREPARING RNN DATA PIPELINE ===")
    vocab, embedding_matrix = build_vocab_and_embeddings(train_data)
    rnn_train_loader = get_rnn_dataloader(train_data, vocab, shuffle=True)
    
    os.makedirs("models/rnn_base", exist_ok=True)
    torch.save(vocab, "models/rnn_base/rnn_vocab.pth")

    # --- Train RNNs ---
    rnn_configs = [
        ("BiLSTM", BiLSTM(100, 128, 14, embedding_matrix)),
        ("BiGRU", BiGRU(100, 128, 14, embedding_matrix))
    ]

    for name, model in rnn_configs:
        print(f"\n=== RNN ({name}) ===")
        metrics[f'{name}_Params'] = count_parameters(model)
        
        model, r_time, r_mem = train_model(model, rnn_train_loader, name, is_transformer=False, epochs=5)
        metrics[f'{name}_Train_Time_s'] = r_time
        metrics[f'{name}_Train_Mem_MB'] = r_mem
        
        save_path = f"models/{name.lower()}"
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{save_path}/{name.lower()}_weights.pth")

        del model; torch.cuda.empty_cache()

    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("\nAll models trained and training metrics saved!")

if __name__ == "__main__":
    main()