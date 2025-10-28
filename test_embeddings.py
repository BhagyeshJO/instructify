from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

sentences = [
    "The sky is blue.",
    "The color of the sky is azure.",
    "I like to play football.",
]

# Return PyTorch tensors directly (bypasses NumPy)
embeddings = model.encode(sentences, convert_to_tensor=True, device='cpu')

# Cosine similarity on tensors
cos_sim = util.cos_sim(embeddings, embeddings)
print(cos_sim)
