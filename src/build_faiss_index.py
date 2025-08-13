import os
import json
import numpy as np
import faiss

# Paths
EMBEDDING_PATH = "data/embeddings/openclip_image_embeddings.npy"
METADATA_PATH = "data/embeddings/image_paths.json"
INDEX_OUTPUT_PATH = "data/embeddings/openclip_index.faiss"

def build_faiss_index():
    # Load embeddings and metadata
    embeddings = np.load(EMBEDDING_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    print(f"Loaded {len(metadata)} image paths")

    # Normalize embeddings for cosine similarity
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(x=embeddings)

    # Create FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product with normalized = cosine
    index.add(embeddings)

    # Save index
    faiss.write_index(index, INDEX_OUTPUT_PATH)
    print(f"✅ FAISS index built and saved to {INDEX_OUTPUT_PATH}")

if __name__ == "__main__":
    build_faiss_index()
