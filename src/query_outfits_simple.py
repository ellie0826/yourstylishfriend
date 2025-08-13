import os
import json
import numpy as np
import faiss
import torch
import open_clip

# Paths
INDEX_PATH = "data/embeddings/openclip_index.faiss"
EMBEDDING_METADATA_PATH = "data/embeddings/image_paths.json"
TOP_K = 5

# Model configuration
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

# Global variables for caching
_model = None
_tokenizer = None
_index = None
_image_paths = None

def load_model():
    """Load and cache the OpenCLIP model"""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print("🔄 Loading OpenCLIP model...")
        _model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        _model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _model = _model.cuda()
            print("✅ Model loaded on GPU")
        else:
            print("✅ Model loaded on CPU")
    
    return _model, _tokenizer

def load_index_and_metadata():
    """Load and cache the FAISS index and metadata"""
    global _index, _image_paths
    if _index is None or _image_paths is None:
        print("🔄 Loading FAISS index and metadata...")
        _index = faiss.read_index(INDEX_PATH)
        with open(EMBEDDING_METADATA_PATH, "r", encoding="utf-8") as f:
            _image_paths = json.load(f)
        print("✅ Index and metadata loaded")
    
    return _index, _image_paths

def get_text_embedding(text):
    """Generate text embedding using cached model"""
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
    """Search for outfits using cached model and index"""
    print(f"\n🔍 Searching for: \"{text_query}\"")
    
    # Use cached index and metadata
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

def interactive_search():
    """Interactive search loop with cached components"""
    # Pre-load everything once
    load_model()
    load_index_and_metadata()
    print("\n🚀 Ready for searches! (Model and index cached)")
    
    while True:
        try:
            query = input("\n📝 Enter your outfit prompt (or 'quit' to exit): ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            results = search_outfits(query)
            show_results(results)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_search()
    print("\n✅ Search completed!")
