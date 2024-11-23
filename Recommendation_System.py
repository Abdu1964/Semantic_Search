from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
from annoy import AnnoyIndex  # For ANN
import re
from nltk.corpus import stopwords, wordnet
import nltk

# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# Preprocessing function to clean text
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Query expansion using synonyms (Minimal)
def expand_query(query, max_synonyms=2):
    expanded_terms = []
    for word in query.split():
        synonyms = wordnet.synsets(word)
        count = 0
        for syn in synonyms:
            for lemma in syn.lemmas():
                if count < max_synonyms:
                    expanded_terms.append(lemma.name())
                    count += 1
                if count >= max_synonyms:
                    break
            if count >= max_synonyms:
                break
    expanded_query = " ".join(set(expanded_terms))
    return f"{query} {expanded_query}"

# Semantic similarity functions
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Fetch news articles using NewsAPI
def fetch_news(api_key, query=None, category=None, num_articles=10):
    base_url = "https://newsapi.org/v2/everything"
    params = {'apiKey': api_key, 'q': query, 'pageSize': num_articles}

    response = requests.get(base_url, params=params)
    print("Request URL:", response.url)  # Debugging: Print the request URL
    print("Response Status Code:", response.status_code)

    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])
        print(f"Fetched {len(articles)} articles.")
        return [
            preprocess(f"{article.get('title', '')}. {article.get('description', '')}")
            for article in articles if article.get('title') and article.get('description')
        ]
    else:
        print(f"Error: {response.json().get('message', 'Unknown error')}")
        return []

# Build ANN index for embeddings
def build_ann_index(embeddings, num_trees=10):
    dimension = len(embeddings[0])  # Dimensionality of embeddings
    ann_index = AnnoyIndex(dimension, 'angular')  # Use angular distance
    for i, embedding in enumerate(embeddings):
        ann_index.add_item(i, embedding)
    ann_index.build(num_trees)  # Build index with specified number of trees
    return ann_index

# Perform the selected semantic search
def perform_search(search_type, query_embedding, article_embeddings, articles, ann_index=None, num_recommendations=5):
    if search_type == "Cosine Similarity":
        scores = [
            (articles[i], cosine_similarity(query_embedding, embedding))
            for i, embedding in enumerate(article_embeddings)
        ]
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)

    elif search_type == "Euclidean Distance":
        scores = [
            (articles[i], euclidean_distance(query_embedding, embedding))
            for i, embedding in enumerate(article_embeddings)
        ]
        ranked = sorted(scores, key=lambda x: x[1])

    elif search_type == "Jaccard Similarity":
        query_tokens = set(query_embedding.split())
        scores = [
            (articles[i], jaccard_similarity(query_tokens, set(article.split())))
            for i, article in enumerate(articles)
        ]
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)

    elif search_type == "ANN":
        nearest_neighbors = ann_index.get_nns_by_vector(query_embedding, num_recommendations, include_distances=True)
        ranked = [(articles[i], 1 - dist) for i, dist in zip(nearest_neighbors[0], nearest_neighbors[1])]

    return ranked[:num_recommendations]

# Main function
def semantic_search(api_key, user_query, search_type, num_recommendations=5):
    model = SentenceTransformer('paraphrase-mpnet-base-v2')

    # Ask the user whether to expand the query or not
    expand_choice = input("Do you want to expand the query with synonyms? (yes/no): ").strip().lower()
    if expand_choice == "yes":
        expanded_query = expand_query(user_query)  # Use the expanded query
    else:
        expanded_query = user_query  # No expansion, use the original query

    # Fetch articles
    articles = fetch_news(api_key, query=expanded_query)
    if not articles:
        print("No articles found. Please try a different query or check your API key.")
        return

    # Generate embeddings for articles and the query
    article_embeddings = model.encode(articles)
    query_embedding = model.encode(preprocess(user_query))

    # Build ANN index if needed
    ann_index = None
    if search_type == "ANN":
        ann_index = build_ann_index(article_embeddings)

    # Perform search
    recommendations = perform_search(
        search_type, query_embedding, article_embeddings, articles, ann_index, num_recommendations
    )

    # Display results
    print(f"\nQuery: '{user_query}' using {search_type}\n")
    print("Top Recommendations:")
    for i, (article, score) in enumerate(recommendations, 1):
        print(f"{i}. {article} (Score: {score:.4f})")

# Execute the script
if __name__ == "__main__":
    API_KEY = "9bb044c8603c4b23a9400e8fabde02be"  # Your API Key
    user_query = input("Enter your query: ")
    print("\nChoose a semantic search technique:")
    print("1. Cosine Similarity")
    print("2. Euclidean Distance")
    print("3. Jaccard Similarity")
    print("4. ANN (Approximate Nearest Neighbors)")
    choice = input("Enter the number of your choice: ")

    search_types = {
        "1": "Cosine Similarity",
        "2": "Euclidean Distance",
        "3": "Jaccard Similarity",
        "4": "ANN"
    }

    search_type = search_types.get(choice, "Cosine Similarity")
    semantic_search(API_KEY, user_query, search_type)
