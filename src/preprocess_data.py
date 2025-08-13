# src/scrape_pinterest_and_tag.py

import os
import json
import requests
import base64
from tqdm import tqdm
from pinscrape import Pinterest
from PIL import Image
import io
import ollama


# --------------------- CONFIG ---------------------
KEYWORDS = [
    "french girl brunch outfit",
    "summer picnic dress",
    "wedding guest fashion 2025",
    "cozy fall streetwear",
    "vacation resort outfits",
    "minimalist chic outfits",
    "bohemian spring outfits",
    "fall outfits 2025",
    "romantic outfits",
    "going out outfits",
    "casual date night outfits",
    "workout outfits",
    "party outfits"
]
OUTPUT_IMAGE_DIR = "output"
OUTPUT_JSON = "data/pinterest_clean.json"
NUM_IMAGES_PER_KEYWORD = 30
NUMBER_OF_WORKERS = 8
# ---------------------------------------------------

def scrape_images():
    p = Pinterest()
    dataset = []
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    for keyword in KEYWORDS:
        print(f"🔍 Scraping: {keyword}")
        urls = p.search(keyword, NUM_IMAGES_PER_KEYWORD)
        folder = os.path.join(OUTPUT_IMAGE_DIR, keyword.replace(" ", "_"))
        p.download(url_list=urls, output_folder=folder, number_of_workers=NUMBER_OF_WORKERS)

        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                dataset.append({
                    "image_path": os.path.join(folder, fname),
                    "text": keyword
                })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ Scraped and saved {len(dataset)} images to {OUTPUT_JSON}")


# def encode_image_to_base64(image_path):
#     try:
#         with Image.open(image_path) as img:
#             if img.mode != 'RGB':
#                 img = img.convert('RGB')
#             img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
#             buffer = io.BytesIO()
#             img.save(buffer, format='JPEG')
#             return base64.b64encode(buffer.getvalue()).decode()
#     except Exception as e:
#         print(f"Error encoding image {image_path}: {e}")
#         return None


# def tag_images_with_ollama_text(model_name="llama3.2:latest"):
#     with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     tagged_data = []
#     for item in tqdm(data, desc="🧠 Tagging with text model"):
#         prompt = f"""
# You are a fashion stylist assistant. Given this Pinterest search keyword:
# "{item['text']}"

# Suggest descriptive fashion-related tags or vibes that would match this search term.
# Focus on:
# - Clothing items and styles
# - Colors and patterns
# - Mood/aesthetic
# - Season/occasion
# - Style categories

# Keep it short (max 6 tags), lowercase, comma-separated.
# Examples: "feminine, summer, light, romantic, casual, french"

# Output only the comma-separated list of tags.
# """
#         try:
#             response = ollama.chat(
#                 model=model_name,
#                 messages=[{'role': 'user', 'content': prompt}]
#             )
#             tags = response['message']['content'].strip()
#             item["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
#             tagged_data.append(item)
#         except Exception as e:
#             print(f"Error tagging {item['image_path']}: {e}")
#             continue

#     with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#         json.dump(tagged_data, f, indent=2)
#     print(f"✅ Tagged and saved {len(tagged_data)} entries to {OUTPUT_JSON}")


# def tag_images_with_ollama_vision(model_name="gemma3:4b"):
#     with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     tagged_data = []
#     for item in tqdm(data, desc="🧠 Tagging with vision model"):
#         image_base64 = encode_image_to_base64(item['image_path'])
#         if not image_base64:
#             continue

#         prompt = f"""
# You are a fashion stylist assistant. Analyze this fashion image and the Pinterest search keyword: "{item['text']}".
# Return 6–8 descriptive fashion-related tags: clothing type, mood, colors, style, season, occasion.

# Output only a comma-separated list. E.g.: "feminine, pastel, picnic, summer, floral, sundress"
# """
#         try:
#             response = ollama.chat(
#                 model=model_name,
#                 messages=[{
#                     'role': 'user',
#                     'content': prompt,
#                     'images': [image_base64]
#                 }]
#             )
#             tags = response['message']['content'].strip()
#             item["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
#             tagged_data.append(item)
#         except Exception as e:
#             print(f"Error tagging {item['image_path']}: {e}")
#             continue

#     with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#         json.dump(tagged_data, f, indent=2)
#     print(f"✅ Tagged and saved {len(tagged_data)} entries to {OUTPUT_JSON}")


if __name__ == "__main__":
    scrape_images()
    # tag_images_with_ollama_vision("gemma3:4b")
    # tag_images_with_ollama_text("llama3:latest")
