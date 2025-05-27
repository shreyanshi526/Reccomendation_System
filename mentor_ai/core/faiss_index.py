import faiss
import numpy as np
import os
import json

# Paths for saving index and ID map
INDEX_PATH = "mentor_ai/core/mentor.index"
ID_MAP_PATH = "mentor_ai/core/mentor_ids.json"

mentor_id_map = []

def build_index(embeddings, mentor_ids):
    embeddings_np = np.array(embeddings).astype('float32')

    if embeddings_np.ndim != 2:
        raise ValueError(f"Expected 2D array for embeddings, got shape {embeddings_np.shape}")

    dim = embeddings_np.shape[1]
    print(f"Building FAISS index with shape: {embeddings_np.shape}, dtype: {embeddings_np.dtype}")

    try:
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings_np)
    except Exception as e:
        print("Error during FAISS index creation:", e)
        raise

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(ID_MAP_PATH, 'w') as f:
        json.dump(mentor_ids, f)

    print(f"FAISS index built and saved with {index.ntotal} vectors.")
    return index

def load_index():

    global mentor_id_map

    if not os.path.exists(INDEX_PATH) or not os.path.exists(ID_MAP_PATH):
        raise FileNotFoundError("Index or ID map not found. Please run build_index() first.")

    try:
        index = faiss.read_index(INDEX_PATH)
    except Exception as e:
        print("Failed to load FAISS index:", e)
        raise

    with open(ID_MAP_PATH, 'r') as f:
        mentor_id_map = json.load(f)

    print(f"FAISS index loaded with {index.ntotal} vectors.")
    return index

def search_similar(query_embedding, top_k=5):

    if not mentor_id_map:
        raise ValueError("mentor_id_map is empty. Load the index first.")

    index = load_index()

    query = np.array([query_embedding]).astype("float32")

    if query.shape[1] != index.d:
        raise ValueError(f"Query dimension mismatch: expected {index.d}, got {query.shape[1]}")

    try:
        distances, indices = index.search(query, top_k)
    except Exception as e:
        print("Error during FAISS search:", e)
        raise

    results = []
    for rank, i in enumerate(indices[0]):
        if i < len(mentor_id_map):
            results.append((mentor_id_map[i], distances[0][rank]))
        else:
            results.append(("UNKNOWN", distances[0][rank]))

    return results
