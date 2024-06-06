import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = AzureOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
)

def get_embeddings(texts):
    response = client.embeddings.create(
        input=texts, model="embedding")
    embeddings = [datum.embedding for datum in response.data]
    return embeddings


def compute_similarity(embeddings):
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


def plot_embeddings(embeddings, texts):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for i, text in enumerate(texts):
        plt.scatter(pca_result[i, 0], pca_result[i, 1])
        plt.text(pca_result[i, 0], pca_result[i, 1], text)
    plt.title('PCA of Embeddings')
    plt.show(block=False)


def cluster_embeddings(embeddings, eps=0.2, min_samples=1):
    clustering = DBSCAN(eps=eps, min_samples=min_samples,
                        metric='cosine').fit(embeddings)
    return clustering.labels_


def group_texts_by_cluster(texts, cluster_labels):
    clustered_texts = defaultdict(list)
    for label, text in zip(cluster_labels, texts):
        clustered_texts[label].append(text)
    return dict(clustered_texts)


# Example usage
texts = ["Walmart", "Walmart Incorporated", "Google",
         "Alphabet Inc.", "Amazon", "Amazon.com, Inc."]
embeddings = get_embeddings(texts)
similarity_matrix = compute_similarity(embeddings)

print("Similarity Matrix:")
print(similarity_matrix)

plot_embeddings(embeddings, texts)

cluster_labels = cluster_embeddings(embeddings, eps=0.2, min_samples=1)
grouped_texts = group_texts_by_cluster(texts, cluster_labels)

print(grouped_texts)
