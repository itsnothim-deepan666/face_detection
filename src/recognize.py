import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recognize_face(name, embedding, threshold=0.5):

    db_embeddings = np.load(f"data/embedding_database/{name}_embeddings.npy")

    similarities = cosine_similarity(embedding.cpu().numpy(), db_embeddings)

    max_similarity = np.max(similarities)

    print(f"Max similarity for {name}: {max_similarity}")
    print(f"Similarities for {name}: {similarities}")

    return max_similarity
