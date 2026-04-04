import os
import urllib.request
import zipfile
import numpy as np
import torch

def get_glove_embeddings(dim=100):
    glove_dir = "glove_data"
    zip_path = f"{glove_dir}/glove.6B.zip"
    txt_path = f"{glove_dir}/glove.6B.{dim}d.txt"
    
    if not os.path.exists(txt_path):
        os.makedirs(glove_dir, exist_ok=True)
        print("Downloading GloVe embeddings...")
        url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(glove_dir)
            
    print("Loading GloVe...")
    embeddings_dict = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

def create_embedding_matrix(vocab, glove_dict, dim=100):
    print("Mapping GloVe vectors to Custom Vocabulary...")
    embedding_matrix = torch.zeros((len(vocab), dim))
    for word, idx in vocab.items():
        if word in glove_dict:
            embedding_matrix[idx] = torch.tensor(glove_dict[word])
        else:
            # Random weights for unknown words
            embedding_matrix[idx] = torch.randn(dim)
    return embedding_matrix