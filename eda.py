import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import Counter
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import os

sns.set_theme(style="whitegrid")
STOP_WORDS = set(ENGLISH_STOP_WORDS)

def clean_and_tokenize(text, remove_stopwords=False):
    """Cleans text and returns a list of words. Optionally removes stopwords."""
    text = str(text).lower()
    # Keep only alphabetical characters
    words = re.findall(r'\b[a-z]{3,}\b', text)
    if remove_stopwords:
        words = [w for w in words if w not in STOP_WORDS]
    return words

def run_full_eda():
    dir = os.path.dirname(os.path.abspath(__file__))
        
    print(f"All graphs will be saved to: {dir}")

    print("\nLoading DBpedia-14 dataset from Hugging Face...")
    dataset = load_dataset("dbpedia_14", split="train")
    df = dataset.to_pandas()

    label_mapping = {
        0: 'Company', 1: 'EducationalInstitution', 2: 'Artist', 3: 'Athlete',
        4: 'OfficeHolder', 5: 'MeanOfTransportation', 6: 'Building', 
        7: 'NaturalPlace', 8: 'Village', 9: 'Animal', 10: 'Plant', 
        11: 'Album', 12: 'Film', 13: 'WrittenWork'
    }
    df['category'] = df['label'].map(label_mapping)
    df['text'] = df['title'] + " " + df['content']

    print(f"Data loaded. Total records: {len(df):,}")

    # 1. CLASS DISTRIBUTION PLOT
    print("\nGenerating Class Distribution Plot...")
    plt.figure(figsize=(12, 6))
    
    category_order = sorted(df['category'].unique())
    sns.countplot(data=df, y='category', order=category_order, palette="mako")
    
    plt.title('DBpedia-14: Class Distribution')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Category')
    plt.savefig(os.path.join(dir, '1_Class_Distribution.png'), bbox_inches='tight')
    plt.close()

    # 2. WORD COUNT DISTRIBUTION
    print("Generating Word Count Distribution...")
    df['word_count'] = df['text'].str.split().str.len()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='word_count', bins=50, color='#667eea', 
                 binrange=(0, df['word_count'].quantile(0.99)))
    plt.title('Word Count Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(dir, '2_Word_Count_Distribution.png'), bbox_inches='tight')
    plt.close()

    # 3. TOKENIZE AND CLEAN TEXT=
    print("Tokenizing and cleaning ALL text...")
    
    df['clean_tokens'] = df['text'].apply(lambda x: clean_and_tokenize(x, remove_stopwords=True))
    df['clean_text'] = df['clean_tokens'].apply(lambda x: " ".join(x))
    all_categories = sorted(df['category'].unique())

    # 4. VOCABULARY RICHNESS
    print("Generating Vocabulary Richness...")
    vocab_stats = []
    for cat in all_categories:
        cat_tokens = [w for tokens in df[df['category'] == cat]['clean_tokens'] for w in tokens]
        vocab_stats.append({'Category': cat, 'Unique Words': len(set(cat_tokens))})
    
    df_vocab = pd.DataFrame(vocab_stats).sort_values(by='Unique Words', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_vocab, x='Category', y='Unique Words', palette="Blues_r")
    plt.title('Vocabulary Richness: Unique Words per Category')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Unique Word Count')
    plt.savefig(os.path.join(dir, '3_Vocabulary_Richness.png'), bbox_inches='tight')
    plt.close()

    # 5. TF-IDF & SIMILARITY
    print("Building TF-IDF Matrix...")
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # --- 5A. Category Similarity Matrix (Heatmap) ---
    print("Generating Category Similarity Matrix...")
    cat_mean_vectors = []
    for cat in all_categories:
        cat_indices = df.index[df['category'] == cat].tolist()
        mean_vec = np.asarray(tfidf_matrix[cat_indices].mean(axis=0)).flatten()
        cat_mean_vectors.append(mean_vec)
        
    similarity_matrix = cosine_similarity(cat_mean_vectors)
    
    plt.figure(figsize=(12, 10)) 
    sns.heatmap(similarity_matrix, xticklabels=all_categories, yticklabels=all_categories, 
                cmap="YlGnBu", annot=True, fmt=".2f", square=True, 
                cbar_kws={'label': 'Cosine Similarity Score'})
    plt.title('Category Similarity Matrix (Cosine Similarity of TF-IDF)', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, '4_Category_Similarity_Matrix.png'), bbox_inches='tight')
    plt.close()

    # --- 5B. TF-IDF Terms ---
    print("Generating TF-IDF Diagram...")
    fig_tf, axes_tf = plt.subplots(7, 2, figsize=(16, 24))
    axes_tf = axes_tf.flatten()

    for i, cat in enumerate(all_categories):
        mean_tfidf = cat_mean_vectors[i] 
        top_indices = mean_tfidf.argsort()[-5:][::-1] 
        
        t_words = feature_names[top_indices]
        t_scores = mean_tfidf[top_indices]
        
        sns.barplot(x=list(t_scores), y=list(t_words), ax=axes_tf[i], palette="magma")
        axes_tf[i].set_title(f"Top TF-IDF Terms: {cat}")
        axes_tf[i].set_xlabel("Mean TF-IDF Score")

    plt.suptitle("Top 5 TF-IDF per Category", fontsize=20, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, '5_TFIDF.png'), bbox_inches='tight')
    plt.close()

    # 6. BIGRAMS
    print("Generating Bigrams Diagram...")
    fig_bg, axes_bg = plt.subplots(7, 2, figsize=(16, 24))
    axes_bg = axes_bg.flatten()

    for i, cat in enumerate(all_categories):
        cat_corpus = df[df['category'] == cat]['clean_text']
        vec = CountVectorizer(ngram_range=(2, 2), max_features=5) 
        bg_matrix = vec.fit_transform(cat_corpus)
        bg_freqs = np.asarray(bg_matrix.sum(axis=0)).flatten()
        bg_words = vec.get_feature_names_out()
        
        sorted_indices = bg_freqs.argsort()[::-1]
        bg_words = bg_words[sorted_indices]
        bg_freqs = bg_freqs[sorted_indices]

        sns.barplot(x=list(bg_freqs), y=list(bg_words), ax=axes_bg[i], palette="rocket")
        axes_bg[i].set_title(f"Top Bigrams: {cat}")
        axes_bg[i].set_xlabel("Frequency")

    plt.suptitle("Top 5 Bigrams per Category", fontsize=20, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, '6_Bigrams.png'), bbox_inches='tight')
    plt.close()

    print("\n EDA Done")

if __name__ == "__main__":
    run_full_eda()