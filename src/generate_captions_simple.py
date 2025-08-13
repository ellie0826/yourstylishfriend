import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Fixed paths and constants
INPUT_JSON = "data/pinterest_clean.json"
OUTPUT_CAPTIONS_JSON = "data/captions/simple_captions.json"
MODEL_ID = "Salesforce/blip-image-captioning-large"
BATCH_SIZE = 4
DEVICE = "cpu"  # Use CPU to avoid compatibility issues
PROMPT = "a detailed description of the outfit including clothing type, color, patterns, materials, shoes, accessories, and style"

print(f"🔄 Loading BLIP model on {DEVICE}...")
try:
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    model.eval()
    print("✅ BLIP model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load BLIP model: {e}")
    print("Trying alternative model...")
    try:
        MODEL_ID = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
        model.eval()
        print("✅ Alternative BLIP model loaded successfully")
    except Exception as e2:
        print(f"❌ Failed to load alternative model: {e2}")
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

print(f"📝 Generating captions for {len(image_paths)} images...")

# Process images one by one to avoid memory issues
for img_path in tqdm(image_paths, desc="Processing images"):
    try:
        # Load and process image
        image = Image.open(img_path).convert("RGB")
        
        # Generate caption with conditional generation
        inputs = processor(image, PROMPT, return_tensors="pt")
        
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=inputs.pixel_values,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,
                num_beams=5,
                early_stopping=True,
                do_sample=False
            )
            
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Clean up the caption - remove the prompt if it appears
        if caption.startswith(PROMPT):
            caption = caption[len(PROMPT):].strip()
        
        # Ensure we have a meaningful caption
        if not caption or len(caption.strip()) < 5:
            caption = "Fashion outfit image"
            
        captions.append({
            "image_path": img_path,
            "caption": caption.strip()
        })
        
    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")
        errors += 1
        captions.append({
            "image_path": img_path,
            "caption": f"(Error: {str(e)})"
        })

if errors > 0:
    print(f"⚠️ {errors} errors occurred during caption generation")

# Save results
os.makedirs(os.path.dirname(OUTPUT_CAPTIONS_JSON), exist_ok=True)
with open(OUTPUT_CAPTIONS_JSON, "w", encoding="utf-8") as f:
    json.dump(captions, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(captions)} captions to {OUTPUT_CAPTIONS_JSON}")
