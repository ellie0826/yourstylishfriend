import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# fixed paths and constants
INPUT_JSON = "data/pinterest_clean.json"
OUTPUT_CAPTIONS_JSON = "data/captions/blip_captions.json"
MODEL_ID = "Salesforce/blip2-opt-2.7b"
BATCH_SIZE = 4 
DEVICE = "cpu"
# DEFAULT_PROMPT = ""  

# # Parse command line arguments
# parser = argparse.ArgumentParser(description='Generate captions for fashion images using BLIP')
# parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, 
#                     help='Custom prompt for image captioning (use short prompts like "a photo of" or leave empty)')
# parser.add_argument('--unconditional', action='store_true',
#                     help='Use unconditional generation (no prompt)')
# args = parser.parse_args()

# PROMPT = args.prompt if not args.unconditional else ""
# print(f"📝 Using prompt: '{PROMPT}' (empty = unconditional generation)")
PROMPT = "What is the outfit of the person in the image? Describe in very specific detail, including clothing type, color, patterns, shoes, accessories, and style."

# Load BLIP
print("🔄 Loading BLIP model...")
processor = Blip2Processor.from_pretrained(MODEL_ID)
model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
if DEVICE != "cpu":
    model = model.to(DEVICE)
model.eval()
print("✅ Model loaded")

# Load scraped image data
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

image_paths = [item["image_path"] for item in data if os.path.exists(item["image_path"])]
captions = []

# Helper function to load and preprocess images
def load_and_process_images(paths):
    images = []
    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
            images.append(image)
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")
    
    if not images:
        return None
    
    # For BLIP conditional generation, use text as starting prompt

    inputs = processor(images=images, text=[PROMPT] * len(images), return_tensors="pt", padding=True)

    
    return inputs.to(DEVICE)

# Generate captions
print(f"📝 Generating captions for {len(image_paths)} images...")
for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i + BATCH_SIZE]
    inputs = load_and_process_images(batch_paths)
    
    if inputs is None:
        continue

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
        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
        
        # Clean up the decoded text - remove the prompt if it appears at the beginning
        cleaned_decoded = []
        for caption in decoded:
            caption = caption[len(PROMPT):].strip()
            print(caption)
            cleaned_decoded.append(caption)
        decoded = cleaned_decoded

    for path, caption in zip(batch_paths, decoded):
        captions.append({
            "image_path": path,
            "caption": caption.strip()
        })

# Save output
os.makedirs(os.path.dirname(OUTPUT_CAPTIONS_JSON), exist_ok=True)
with open(OUTPUT_CAPTIONS_JSON, "w", encoding="utf-8") as f:
    json.dump(captions, f, indent=2)

print(f"✅ Saved {len(captions)} captions to {OUTPUT_CAPTIONS_JSON}")
