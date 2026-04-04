from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

def get_transformer_components(model_id, num_labels=14):
    print(f"Loading {model_id} components...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    return tokenizer, model

def prepare_transformer_data(dataset, tokenizer, batch_size=32, shuffle=True):
    def tokenize(batch):
        return tokenizer(
            batch['content'], 
            padding='max_length', 
            truncation=True, 
            max_length=100
        )
    
    print("Tokenizing data for Transformer...")
    encodings = dataset.map(tokenize, batched=True)
    encodings.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    loader = DataLoader(encodings, batch_size=batch_size, shuffle=shuffle)
    return loader