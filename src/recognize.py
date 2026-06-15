import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def recognize_face(name, embedding, threshold=0.5):

    path = f"data/embedding_database/{name}_embeddings.npy"
    if not os.path.exists(path):
        print(f"No embeddings found for {name}. Please register the person first.")
        return None
    
    db_embeddings = np.load(f"data/embedding_database/{name}_embeddings.npy")

    similarities = cosine_similarity(embedding.cpu().numpy(), db_embeddings)

    max_similarity = np.max(similarities)

    print(f"Max similarity for {name}: {max_similarity}")
    print(f"Similarities for {name}: {similarities}")

    return max_similarity
