import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch

try:
    from promptcap import PromptCap
except ImportError:
    print("❌ PromptCap not installed. Please install it with: pip install promptcap")
    exit(1)

# fixed paths and constants
INPUT_JSON = "data/pinterest_clean.json"
OUTPUT_CAPTIONS_JSON = "data/captions/promptcap_captions.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PROMPT = "Please describe the outfit in this image in very specific detail, including what clothing type, color, patterns, materials, shoes, accessories, and style it is. Be very specific and detailed in your description."

print(f"🔄 Loading PromptCap model on {DEVICE}...")
try:
    model = PromptCap("tifa-benchmark/promptcap-coco-vqa")
    if torch.cuda.is_available():
        model.cuda()
    print("✅ PromptCap model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load PromptCap model: {e}")
    exit(1)

# Load scraped image data
print("📂 Loading image data...")
try:
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ Input file not found: {INPUT_JSON}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"❌ Invalid JSON in input file: {e}")
    exit(1)

# Filter existing image paths
image_paths = [item["image_path"] for item in data if os.path.exists(item["image_path"])]
missing_images = len(data) - len(image_paths)
if missing_images > 0:
    print(f"⚠️ {missing_images} images not found on disk, processing {len(image_paths)} images")

captions = []
errors = 0

print(f"📝 Generating PromptCap captions for {len(image_paths)} images...")

for img_path in tqdm(image_paths, desc="Processing images"):
    try:
        cap = model.caption(PROMPT, img_path)
        if not cap or cap.strip() == "":
            cap = "(Empty caption generated)"
    except Exception as e:
        cap = f"(Error: {str(e)})"
        errors += 1
    
    captions.append({"image_path": img_path, "caption": cap})

if errors > 0:
    print(f"⚠️ {errors} errors occurred during caption generation")

# Save results
os.makedirs(os.path.dirname(OUTPUT_CAPTIONS_JSON), exist_ok=True)
with open(OUTPUT_CAPTIONS_JSON, "w", encoding="utf-8") as f:
    json.dump(captions, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(captions)} PromptCap captions to {OUTPUT_CAPTIONS_JSON}")
