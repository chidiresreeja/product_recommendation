import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("nlp.csv")
df['title'] = df['title'].fillna("").str.lower()

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all product titles into vectors (once)
product_titles = df['title'].tolist()
title_embeddings = model.encode(product_titles, show_progress_bar=True)

# Recommend similar products using semantic similarity
def recommend(query):
    query_vector = model.encode([query.lower()])
    similarities = cosine_similarity(query_vector, title_embeddings)[0]
    top_indices = similarities.argsort()[-5:][::-1]
    return df.iloc[top_indices][['title', 'stars', 'price']].to_dict(orient='records')
