import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
from glove import get_glove_embeddings, create_embedding_matrix

SEQ_LEN = 100

# --- BASE CLASS ---
class BaseRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes, embedding_matrix):
        super(BaseRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

# --- CHILD CLASSES ---
class BiLSTM(BaseRNN):
    def __init__(self, embed_dim, hidden_dim, num_classes, embedding_matrix):
        super().__init__(embed_dim, hidden_dim, num_classes, embedding_matrix)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        final_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(final_hidden)

class BiGRU(BaseRNN):
    def __init__(self, embed_dim, hidden_dim, num_classes, embedding_matrix):
        super().__init__(embed_dim, hidden_dim, num_classes, embedding_matrix)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        final_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(final_hidden)

def tokenize(text):
    return re.findall(r'\w+|[^\w\s]', text.lower())

def build_vocab_and_embeddings(train_data, dim=100):
    print("Building Custom Vocabulary...")
    vocab = {"<pad>": 0, "<unk>": 1}
    
    for item in train_data:
        tokens = tokenize(item['content'][:512])
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
                
    print(f"Vocabulary Size: {len(vocab)} words")
    
    glove_dict = get_glove_embeddings(dim=dim)
    embedding_matrix = create_embedding_matrix(vocab, glove_dict, dim=dim)
    return vocab, embedding_matrix

def get_rnn_dataloader(dataset, vocab, batch_size=32, shuffle=False):
    def collate_batch(batch):
        labels = torch.tensor([item['label'] for item in batch])
        
        batch_texts = []
        for item in batch:
            tokens = tokenize(item['content'][:512])
            token_ids = [vocab.get(t, 1) for t in tokens[:SEQ_LEN]]
            batch_texts.append(torch.tensor(token_ids))
            
        texts = pad_sequence(batch_texts, batch_first=True, padding_value=vocab['<pad>'])
        
        if texts.size(1) < SEQ_LEN:
            padding = torch.full((texts.size(0), SEQ_LEN - texts.size(1)), vocab['<pad>'], dtype=torch.long)
            texts = torch.cat([texts, padding], dim=1)
        elif texts.size(1) > SEQ_LEN:
            texts = texts[:, :SEQ_LEN]
            
        return {'input_ids': texts, 'label': labels}

    print("Creating RNN DataLoader...")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)
    return loader