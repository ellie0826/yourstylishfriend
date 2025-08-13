import os
import json
import numpy as np
import faiss
import time
import ollama
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# Paths
INDEX_PATH = "data/embeddings/fashionclip_index.faiss"
EMBEDDING_METADATA_PATH = "data/embeddings/image_paths.json"
TOP_K = 3

# Global variables for model caching
_processor = None
_model = None

def load_model():
    global _processor, _model
    if _processor is None or _model is None:
        print("🔄 Loading FashionCLIP model...")
        _processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        _model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")

        # Check for MPS support (Apple GPU)
        if torch.backends.mps.is_available():
            _model.to("mps")
            print("✅ FashionCLIP model loaded on Apple GPU (MPS)")
        else:
            _model.to("cpu")
            print("⚠️ MPS not available. Model loaded on CPU")
        
        _model.eval()
    return _processor, _model



def load_index_and_metadata():
    index = faiss.read_index(INDEX_PATH)
    with open(EMBEDDING_METADATA_PATH, "r", encoding="utf-8") as f:
        image_paths = json.load(f)
    return index, image_paths

def get_text_embedding(text):
    processor, model = load_model()
    inputs = processor(text=[text], return_tensors="pt")

    # Move inputs to same device as model (e.g., MPS)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

    embedding = outputs.detach().to("cpu").squeeze().float().numpy()
    embedding = embedding / np.linalg.norm(embedding)  # Normalize
    return embedding.astype('float32')


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

import base64
import io

def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail((1024, 1024))  # Reduce size for efficiency
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        print(f"⚠️ Error encoding image {image_path}: {e}")
        return None

# def generate_explanations_with_llm(query, results):
#     print("\n💬 Generating explanations with LLM...")

#     explanations = []
#     for result in results:
#         keyword = os.path.basename(os.path.dirname(result['image_path'])).replace("_", " ")

#         prompt = f"""
# You are a fashion stylist. A user asked: "{query}"

# One of the recommended outfits is based on the Pinterest keyword: "{keyword}".

# Write 1–2 short sentences to explain **why** this outfit is a good match for the user query, considering occasion, style, and aesthetic.
# Avoid generic explanations. Be specific and use fashion language if possible.
# """

#         try:
#             response = ollama.chat(
#                 model="llama3",  # Or "mistral" or any other local model
#                 messages=[{'role': 'user', 'content': prompt}]
#             )
#             explanation = response['message']['content'].strip()
#         except Exception as e:
#             explanation = f"(Error generating explanation: {e})"

#         result["explanation"] = explanation
#         explanations.append(explanation)

#     return results

def generate_explanations_with_llm(query, results):
    print("\n🧠 Generating image-based LLM explanations...")

    for result in results:
        image_base64 = encode_image_to_base64(result['image_path'])
        if not image_base64:
            result["explanation"] = "(Image could not be processed)"
            continue

        prompt = f"""
A user asked: "{query}"

You are a fashion stylist assistant. Analyze the image and explain in 1–2 short sentences why this outfit is a good match for the query.
Focus on style, occasion, color palette, vibe, and clothing items. Avoid vague or generic answers.
Be descriptive and insightful, using fashion language.
"""

        try:
            response = ollama.chat(
                model="llava:latest",  # Vision-capable model
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_base64]
                }]
            )
            explanation = response['message']['content'].strip()
        except Exception as e:
            explanation = f"(Error: {e})"

        result["explanation"] = explanation

    return results


def show_results(results):
    print(f"\n🎯 Top matches:")
    for item in results:
        print(f"{item['rank']}. {item['image_path']} — Similarity: {item['similarity_score']:.3f}")
        if "explanation" in item:
            print(f"   💬 {item['explanation']}")


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

                # Title with rank and similarity
                ax.set_title(f"Rank {item['rank']}\nScore: {item['similarity_score']:.3f}", fontsize=10)

                # Explanation displayed below image
                explanation = item.get("explanation", "")
                ax.text(0.5, -0.15, explanation, fontsize=8, ha='center', va='top', transform=ax.transAxes, wrap=True)

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

    results = generate_explanations_with_llm(query, results)

    show_results(results)
    print("✅ Display done")
