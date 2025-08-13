#!/usr/bin/env python3
"""
Command-line version of query_outfits.py that takes a prompt as an argument
Usage: python query_outfits_cli.py "your outfit prompt here"
"""

import os
import json
import numpy as np
import faiss
import torch
import sys
import argparse
from PIL import Image
import open_clip

# Paths
INDEX_PATH = "../data/embeddings/openclip_index.faiss"
EMBEDDING_METADATA_PATH = "../data/embeddings/image_paths.json"
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
    """Load FAISS index and metadata"""
    index = faiss.read_index(INDEX_PATH)
    with open(EMBEDDING_METADATA_PATH, "r", encoding="utf-8") as f:
        image_paths = json.load(f)
    return index, image_paths

def get_text_embedding(text):
    """Generate text embedding using OpenCLIP"""
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

def search_outfits(text_query, top_k=TOP_K):
    """Search for outfits matching the text query"""
    print(f"🔍 Searching for: \"{text_query}\"")
    
    # Load index and metadata
    index, image_paths = load_index_and_metadata()
    
    # Generate query embedding
    query_vec = get_text_embedding(text_query)
    
    # Search in FAISS index
    distances, indices = index.search(query_vec[np.newaxis, :], top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'image_path': image_paths[idx],
            'similarity_score': distances[0][i],
            'rank': i + 1
        })
    return results

def show_results(results):
    """Display search results"""
    print(f"\n🎯 Top {len(results)} matches:")
    for item in results:
        print(f"{item['rank']}. {item['image_path']} — Similarity: {item['similarity_score']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Search for fashion outfits using text prompts')
    parser.add_argument('prompt', help='Text description of the outfit you want to find')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return (default: 5)')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    
    args = parser.parse_args()
    
    try:
        # Search for outfits
        results = search_outfits(args.prompt, args.top_k)
        
        if args.json:
            # Output as JSON
            import json
            print(json.dumps(results, indent=2))
        else:
            # Output as formatted text
            show_results(results)
            
        print(f"\n✅ Found {len(results)} matching outfits")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Required data files not found. {e}")
        print("💡 Make sure the embedding files exist in data/embeddings/")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during search: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
