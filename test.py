import torch
import time
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from rnn import BiLSTM, BiGRU, tokenize as rnn_tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 100

CLASS_NAMES = [
    "Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
    "Mean Of Transportation", "Building", "Natural Place", "Village", "Animal",
    "Plant", "Album", "Film", "Written Work"
]

def run_inference(model, texts, tokenizer=None, rnn_vocab=None, is_transformer=True, batch_size=64):
    model = model.to(device)
    model.eval()
    preds = []
    
    start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for i in tqdm(range(0, len(texts), batch_size), desc="Inferencing"):
        batch_texts = texts[i : i + batch_size]
        
        if is_transformer:
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=SEQ_LEN, padding=True).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            
        else:
            batch_inputs = []
            for text in batch_texts:
                tokens = rnn_tokenize(text)
                token_ids = [rnn_vocab.get(t, 1) for t in tokens[:SEQ_LEN]]
                if len(token_ids) < SEQ_LEN:
                    token_ids += [0] * (SEQ_LEN - len(token_ids))
                batch_inputs.append(token_ids)
                
            input_tensor = torch.tensor(batch_inputs).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())

    inf_time = time.perf_counter() - start_time
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    return preds, inf_time, max_memory

def save_confusion_matrix(true_labels, preds, model_name):
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_confusion_matrix.png", dpi=300)
    plt.close()

def main():
    print(f"Using device: {device}")
    
    raw_dataset = load_dataset("dbpedia_14")
    # Using 7,000 => exactly 500 per class
    test_dataset = raw_dataset['test'].train_test_split(train_size=7000, stratify_by_column="label")['train']
    texts = test_dataset['content']
    true_labels = test_dataset['label']

    results = []
    all_preds = {} # Dictionary to store predictions for error analysis

    # --- Evaluate Transformers ---
    transformer_models = ["distilbert", "roberta"]
    for model_name in transformer_models:
        display_name = "DistilBERT" if model_name == "distilbert" else "RoBERTa"
        print(f"\nEvaluating {display_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(f"models/{model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(f"models/{model_name}")
        
        preds, inf_time, inf_mem = run_inference(model, texts, tokenizer=tokenizer, is_transformer=True)
        all_preds[display_name] = preds
        
        save_confusion_matrix(true_labels, preds, display_name)
        
        results.append({
            "Model": display_name,
            "Accuracy": accuracy_score(true_labels, preds),
            "Macro F1": f1_score(true_labels, preds, average='macro'),
            "Inference Time (s)": inf_time,
            "Inference Memory (MB)": inf_mem
        })
        del model; torch.cuda.empty_cache()

    # --- Evaluate RNNs ---
    print("\nLoading RNN Shared Data...")
    rnn_vocab = torch.load("models/rnn_base/rnn_vocab.pth", weights_only=False)
    dummy_emb = torch.zeros((len(rnn_vocab), 100))

    rnn_configs = [
        ("BiLSTM", BiLSTM(100, 128, 14, dummy_emb), "models/bilstm/bilstm_weights.pth"),
        ("BiGRU", BiGRU(100, 128, 14, dummy_emb), "models/bigru/bigru_weights.pth")
    ]

    for name, model, path in rnn_configs:
        print(f"\nEvaluating {name}...")
        model.load_state_dict(torch.load(path, weights_only=True))
        preds, inf_time, inf_mem = run_inference(model, texts, rnn_vocab=rnn_vocab, is_transformer=False)
        all_preds[name] = preds
        
        save_confusion_matrix(true_labels, preds, name)
        
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(true_labels, preds),
            "Macro F1": f1_score(true_labels, preds, average='macro'),
            "Inference Time (s)": inf_time,
            "Inference Memory (MB)": inf_mem
        })
        del model; torch.cuda.empty_cache()

    # --- Generate Error Analysis ---
    print("\nGenerating Error Analysis (Hard Examples)...")
    errors = []
    for i in range(len(texts)):
        actual = true_labels[i]
        # Check if any model got it wrong
        if any(all_preds[m][i] != actual for m in all_preds):
            error_dict = {
                "Text": texts[i],
                "Actual Label": CLASS_NAMES[actual]
            }
            for m in all_preds:
                error_dict[f"{m} Guessed"] = CLASS_NAMES[all_preds[m][i]]
            
            errors.append(error_dict)
            if len(errors) >= 50:
                break

    # Save to CSV
    csv_file = "error_analysis_hard_examples.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        fieldnames = ["Text", "Actual Label"] + [f"{m} Guessed" for m in all_preds]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    print(f"-> Saved {len(errors)} hard examples to '{csv_file}'")

    with open("training_metrics.json", "r") as f:
        train_metrics = json.load(f)

    for row in results:
        m_name = row["Model"]
        row["Learnable Params"] = train_metrics.get(f"{m_name}_Params", 0)
        row["Train Time (s)"] = train_metrics.get(f"{m_name}_Train_Time_s", 0)
        row["Train Memory (MB)"] = train_metrics.get(f"{m_name}_Train_Mem_MB", 0)

    df = pd.DataFrame(results)
    df.to_csv("backbone_comparison_metrics.csv", index=False)
    
    print("\n--- FINAL EVALUATION SUMMARY ---")
    print(df.to_string(index=False))
    print("\n-> Saved comprehensive comparison to 'backbone_comparison_metrics.csv'")

if __name__ == "__main__":
    main()