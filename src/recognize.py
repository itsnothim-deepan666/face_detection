import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


def recognize_face(name, embedding, threshold=0.6):
    path = f"data/embeddings/{name}_embeddings.npy"

    if not os.path.exists(path):
        print(f"No embeddings found for {name}. Register first.")
        return None

    db_embeddings = np.load(path)

    # ensure query shape (1,512)
    query = np.array(embedding).reshape(1, -1)

    # normalize query
    query = query / (np.linalg.norm(query) + 1e-10)

    # ===== CENTROID MATCHING =====
    centroid = np.mean(db_embeddings, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

    centroid_score = cosine_similarity(
        query,
        centroid.reshape(1, -1)
    )[0][0]

    # ===== TOP-K SAMPLE CHECK =====
    sample_scores = cosine_similarity(query, db_embeddings)[0]
    topk = np.sort(sample_scores)[-3:]
    topk_score = np.mean(topk)

    # combine both
    final_score = (centroid_score + topk_score) / 2.0

    print(f"Centroid score: {centroid_score:.4f}")
    print(f"Top-k score: {topk_score:.4f}")
    print(f"Final score: {final_score:.4f}")

    if final_score < threshold:
        return "Unknown", final_score

    return name, final_score