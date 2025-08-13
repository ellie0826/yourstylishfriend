# src/embed_captions.py
import json, numpy as np
from sentence_transformers import SentenceTransformer

captions = json.load(open("data/captions/captions.json"))
texts, paths = [], []
for row in captions:
  c = row["caption"]
  if isinstance(c, dict) and "raw_caption" not in c:
    # join fields into one string
    ctext = ", ".join([f"{k}: {v}" for k,v in c.items()])
  else:
    ctext = c.get("raw_caption", str(c))
  texts.append(ctext)
  paths.append(row["image_path"])
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") 
emb = model.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
np.save("data/embeddings/caption_emb.npy", emb.astype("float32"))
json.dump(paths, open("data/embeddings/caption_paths.json","w"))
