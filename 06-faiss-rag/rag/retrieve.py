import numpy as np
import faiss


def search_index(index, query_embedding: list[float], documents: list[str], top_k: int = 3):
    # Convert query list to a 2D NumPy array and ensure it's in float32 format
    query_vector = np.array([query_embedding]).astype("float32")

    # faiss.normalize_L2(query_vector)
    
    # Query the index for the top_k most similar vectors; returns distances and their row indices
    distances, indices = index.search(query_vector, top_k)

    results = []
    # Loop through the first (and only) row of results to map indices back to document text
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            # Use the index ID to look up the original string from your documents list
            "text": documents[idx],
            # Store the similarity distance (smaller L2 distance = more similar)
            "distance": float(dist)
        })

    return results
