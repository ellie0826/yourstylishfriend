# src/embedding_openclip.py

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip

# --------------------- CONFIG ---------------------
INPUT_JSON = "data/pinterest_clean.json"
OUTPUT_EMBEDDINGS = "data/embeddings/openclip_image_embeddings.npy"
OUTPUT_IMAGE_PATHS = "data/embeddings/image_paths.json"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"  # You can change this to other models like "ViT-L-14", "ViT-H-14", etc.
PRETRAINED = "openai"    # You can use "laion2b_s34b_b79k" or other pretrained weights
# --------------------------------------------------

# Load OpenCLIP model
print(f"🔄 Loading OpenCLIP model: {MODEL_NAME} with {PRETRAINED} weights...")
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model.to(DEVICE)
model.eval()

# Use the validation preprocessing for inference
# If preprocess_val is a tuple, use the first element (which should be the transform)
if isinstance(preprocess_val, tuple):
    preprocess = preprocess_val[0]
else:
    preprocess = preprocess_val

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
            image_tensor = preprocess(image)
            images.append(image_tensor)
        except Exception as e:
            print(f"Skipping {path}: {e}")
    
    if images:
        return torch.stack(images).to(DEVICE)
    else:
        return torch.empty(0, 3, 224, 224).to(DEVICE)  # Empty tensor with correct shape

# Embed images in batches
print(f"📸 Embedding {len(image_paths)} images with OpenCLIP...")
for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i + BATCH_SIZE]
    image_batch = load_and_process_images(batch_paths)
    
    if image_batch.size(0) > 0:  # Only process if we have valid images
        with torch.no_grad():
            image_features = model.encode_image(image_batch)
            # Normalize the features for better similarity search
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().numpy()
            embeddings.append(image_features)

# Combine all batches and save
if embeddings:
    embeddings = np.vstack(embeddings)
    os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS), exist_ok=True)
    np.save(OUTPUT_EMBEDDINGS, embeddings)

    with open(OUTPUT_IMAGE_PATHS, "w", encoding="utf-8") as f:
        json.dump(image_paths, f, indent=2)

    print(f"✅ Saved {len(embeddings)} image embeddings to {OUTPUT_EMBEDDINGS}")
    print(f"📊 Embedding shape: {embeddings.shape}")
else:
    print("❌ No valid embeddings were generated")
