# save as download_model.py
from sentence_transformers import SentenceTransformer

print("Downloading model... This may take 2-3 minutes...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model downloaded successfully!")
print(f"Saved to: {model.cache_folder}")