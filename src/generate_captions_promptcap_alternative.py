import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Fixed paths and constants
INPUT_JSON = "data/pinterest_clean.json"
OUTPUT_CAPTIONS_JSON = "data/captions/promptcap_alternative_captions.json"
MODEL_ID = "Salesforce/blip2-opt-2.7b"
BATCH_SIZE = 2  # Smaller batch size for larger model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "Question: Please describe the outfit in this image in very specific detail, including what clothing type, color, patterns, materials, shoes, accessories, and style it is. Be very specific and detailed in your description. Answer:"

print(f"🔄 Loading BLIP-2 model on {DEVICE}...")
try:
    processor = Blip2Processor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    model.eval()
    print("✅ BLIP-2 model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load BLIP-2 model: {e}")
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

# Helper function to load and process images
def load_and_process_images(paths):
    images = []
    valid_paths = []
    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
            images.append(image)
            valid_paths.append(path)
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")
    
    if not images:
        return None, []
    
    # Process images with prompt
    inputs = processor(images=images, text=[PROMPT] * len(images), return_tensors="pt", padding=True)
    
    return inputs.to(DEVICE), valid_paths

print(f"📝 Generating BLIP-2 captions for {len(image_paths)} images...")

# Process images in batches
for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Processing batches"):
    batch_paths = image_paths[i:i + BATCH_SIZE]
    inputs, valid_paths = load_and_process_images(batch_paths)
    
    if inputs is None:
        continue

    try:
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=inputs.pixel_values,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                num_beams=5,
                early_stopping=True,
                do_sample=False
            )
            decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
            
            # Clean up the decoded text - remove the prompt if it appears at the beginning
            cleaned_decoded = []
            for caption in decoded:
                # Remove the prompt from the beginning if it exists
                if caption.startswith(PROMPT):
                    caption = caption[len(PROMPT):].strip()
                # Also try to remove just the "Answer:" part
                if caption.startswith("Answer:"):
                    caption = caption[7:].strip()
                cleaned_decoded.append(caption)
            
            for path, caption in zip(valid_paths, cleaned_decoded):
                captions.append({
                    "image_path": path,
                    "caption": caption.strip() if caption.strip() else "(Empty caption generated)"
                })
                
    except Exception as e:
        print(f"⚠️ Error processing batch: {e}")
        errors += 1
        # Add error entries for this batch
        for path in valid_paths:
            captions.append({
                "image_path": path,
                "caption": f"(Error: {str(e)})"
            })

if errors > 0:
    print(f"⚠️ {errors} batch errors occurred during caption generation")

# Save results
os.makedirs(os.path.dirname(OUTPUT_CAPTIONS_JSON), exist_ok=True)
with open(OUTPUT_CAPTIONS_JSON, "w", encoding="utf-8") as f:
    json.dump(captions, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(captions)} BLIP-2 captions to {OUTPUT_CAPTIONS_JSON}")
