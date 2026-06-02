import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

from google import genai
from dotenv import load_dotenv
import os

# ==========================
# Gemini Setup
# ==========================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ==========================
# Load FAISS Index
# ==========================

index = faiss.read_index(
    "faiss_index.bin"
)

print(
    f"Loaded Index with {index.ntotal} vectors"
)

# ==========================
# Load Chunks
# ==========================

with open(
    "chunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)

print(
    f"Loaded {len(chunks)} chunks"
)

# ==========================
# Load Embedding Model
# ==========================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==========================
# User Question
# ==========================

question = input(
    "\nAsk a Question: "
)

# ==========================
# Query Embedding
# ==========================

query_embedding = model.encode(
    [question]
)

query_embedding = np.array(
    query_embedding,
    dtype=np.float32
)

# ==========================
# Retrieval
# ==========================

k = 3

distances, indices = index.search(
    query_embedding,
    k
)

# ==========================
# Context
# ==========================

context = ""

for idx in indices[0]:
    context += chunks[idx]
    context += "\n\n"

print("\nRetrieved Context:\n")
print(context)

# ==========================
# Prompt
# ==========================

prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not present,
say:

"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

# ==========================
# Gemini
# ==========================

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

# ==========================
# Answer
# ==========================

print("\n===================")
print("ANSWER")
print("===================\n")

print(response.text)