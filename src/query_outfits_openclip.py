import os
import json
import numpy as np
import faiss
import torch
from PIL import Image
import matplotlib.pyplot as plt
import open_clip
print(torch.backends.mps.is_available())  # Should be True


# Paths
INDEX_PATH = "data/embeddings/openclip_index.faiss"
EMBEDDING_METADATA_PATH = "data/embeddings/image_paths.json"
TOP_K = 5

# Model configuration
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

# Global variables for model caching
_model = None
_tokenizer = None

def load_model():
    """Load and cache the OpenCLIP model"""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print("🔄 Loading OpenCLIP model...")
        _model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _model = _model.cuda()
            print("✅ OpenCLIP model loaded on GPU")
        else:
            print("✅ OpenCLIP model loaded on CPU")
        
        _model.eval()
    return _model, _tokenizer

def load_index_and_metadata():
    index = faiss.read_index(INDEX_PATH)
    with open(EMBEDDING_METADATA_PATH, "r", encoding="utf-8") as f:
        image_paths = json.load(f)
    return index, image_paths

def get_text_embedding(text):
    model, tokenizer = load_model()
    
    # Tokenize the text
    text_tokens = tokenizer([text])
    
    # Move inputs to GPU if model is on GPU
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        text_tokens = text_tokens.cuda()
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # Normalize the features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    embedding = text_features.squeeze().cpu().numpy()
    return embedding.astype('float32')

import time

def search_outfits(text_query, top_k=TOP_K):
    print(f"\n🔍 Searching for: \"{text_query}\"")
    start = time.time()

    index, image_paths = load_index_and_metadata()
    print(f"⏱️ Loaded index and metadata in {time.time() - start:.2f}s")

    query_vec = get_text_embedding(text_query)
    print(f"⏱️ Embedded query in {time.time() - start:.2f}s")

    distances, indices = index.search(query_vec[np.newaxis, :], top_k)
    print(f"⏱️ FAISS search done in {time.time() - start:.2f}s")

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'image_path': image_paths[idx],
            'similarity_score': distances[0][i],
            'rank': i + 1
        })
    return results

def show_results(results):
    print(f"\n🎯 Top matches:")
    for item in results:
        print(f"{item['rank']}. {item['image_path']} — Similarity: {item['similarity_score']:.3f}")

    # Display the images (optional)
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, item in zip(axes, results):
        try:
            if os.path.exists(item["image_path"]):
                img = Image.open(item["image_path"])
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"Rank {item['rank']}\nScore: {item['similarity_score']:.3f}", fontsize=10)
            else:
                ax.text(0.5, 0.5, f"Image not found:\n{item['image_path']}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis("off")
        except Exception as e:
            print(f"⚠️ Could not display image: {item['image_path']} — {e}")
            ax.text(0.5, 0.5, f"Error loading image", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    query = input("📝 Enter your outfit prompt: ")
    print("✅ Query received")

    results = search_outfits(query)
    print("✅ Search complete")

    show_results(results)
    print("✅ Display done")
