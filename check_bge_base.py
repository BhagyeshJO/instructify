from time import time
from sentence_transformers import SentenceTransformer
import torch, platform, psutil, os

model_name = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(model_name, device="cpu")

texts = [f"Sample sentence {i} about warranty and returns." for i in range(200)]
t0 = time()
emb = model.encode(texts, batch_size=32, convert_to_tensor=True, device="cpu", normalize_embeddings=True)
t1 = time()

per_100 = (t1 - t0) / (len(texts) / 100)
print(f"Model: {model_name}")
print(f"Python: {platform.python_version()} | Torch: {torch.__version__}")
print(f"Time per 100 embeddings (approx): {per_100:.2f} sec")
print("CUDA available:", torch.cuda.is_available())

# show simple RAM info
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"RAM total: {mem.total/1e9:.1f} GB | available: {mem.available/1e9:.1f} GB")
except Exception:
    pass
