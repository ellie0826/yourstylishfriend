# src/embed_fashionclip.py

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# --------------------- CONFIG ---------------------
INPUT_JSON = "data/pinterest_clean.json"
OUTPUT_EMBEDDINGS = "data/embeddings/fclip_image_embeddings.npy"
OUTPUT_IMAGE_PATHS = "data/embeddings/image_paths.json"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------

# Load FashionCLIP model
print("🔄 Loading FashionCLIP model...")
processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")
model.to(DEVICE)
model.eval()

# Load scraped image data
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

image_paths = [item["image_path"] for item in data if os.path.exists(item["image_path"])]
embeddings = []

# Helper function to preprocess a batch of images
def load_and_process_images(paths):
    images = []
    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
            images.append(image)
        except Exception as e:
            print(f"Skipping {path}: {e}")
    return processor(images=images, return_tensors="pt", padding=True).to(DEVICE)

# Embed images in batches
print(f"📸 Embedding {len(image_paths)} images with FashionCLIP...")
for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i + BATCH_SIZE]
    inputs = load_and_process_images(batch_paths)

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        outputs = outputs.cpu().numpy()
        embeddings.append(outputs)

# Combine all batches and save
embeddings = np.vstack(embeddings)
os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS), exist_ok=True)
np.save(OUTPUT_EMBEDDINGS, embeddings)

with open(OUTPUT_IMAGE_PATHS, "w", encoding="utf-8") as f:
    json.dump(image_paths, f, indent=2)

print(f"✅ Saved {len(embeddings)} image embeddings to {OUTPUT_EMBEDDINGS}")