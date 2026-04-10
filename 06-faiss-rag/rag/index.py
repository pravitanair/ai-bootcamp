import faiss
import numpy as np

def build_faiss_index(embeddings: list[list[float]]):
    # Convert list to a NumPy array and ensure it's in float32 format for Faiss compatibility
    vectors = np.array(embeddings).astype("float32")

    
    
    # Identify the length (dimensionality) of a single vector (e.g., 768 or 1536)
    dimension = vectors.shape[1]
    
    # Initialize a "flat" index that uses Euclidean (L2) distance for exact, brute-force matching
    index = faiss.IndexFlatL2(dimension)

    # let's try flat IP

    # faiss.normalize_L2(vectors)

    # index = faiss.IndexFlatIP(dimension)
    
    # Load the vectors into the index's memory so they are ready to be searched
    index.add(vectors)
    
    return index
