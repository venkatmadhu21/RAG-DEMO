from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from google import genai
from dotenv import load_dotenv
import os

# =====================================
# Load Environment Variables
# =====================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# =====================================
# Load PDF
# =====================================

reader = PdfReader("data/sample.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

# =====================================
# Chunking
# =====================================

chunk_size = 300
chunk_overlap = 50

chunks = []

for i in range(0, len(text), chunk_size - chunk_overlap):
    chunk = text[i:i + chunk_size]

    if chunk.strip():
        chunks.append(chunk)

print(f"\nCreated {len(chunks)} chunks")

# =====================================
# Embedding Model
# =====================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chunk_embeddings = model.encode(chunks)

chunk_embeddings = np.array(
    chunk_embeddings,
    dtype=np.float32
)

# =====================================
# Create FAISS Index
# =====================================

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(chunk_embeddings)

print(f"Vectors Stored: {index.ntotal}")

# =====================================
# User Query
# =====================================

question = input("\nAsk a Question: ")

# =====================================
# Query Embedding
# =====================================

query_embedding = model.encode([question])

query_embedding = np.array(
    query_embedding,
    dtype=np.float32
)

# =====================================
# Retrieval
# =====================================

k = 3

distances, indices = index.search(
    query_embedding,
    k
)

print("\n===================")
print("RETRIEVAL RESULTS")
print("===================")

print("\nDistances:")
print(distances)

print("\nIndices:")
print(indices)

# =====================================
# Build Context
# =====================================

context = ""

for idx in indices[0]:
    context += chunks[idx]
    context += "\n\n"

print("\n===================")
print("RETRIEVED CONTEXT")
print("===================\n")

print(context)

# =====================================
# Prompt Construction
# =====================================

prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not available in the context,
respond with:

"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

# =====================================
# Gemini Generation
# =====================================

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

# =====================================
# Final Answer
# =====================================

print("\n===================")
print("ANSWER")
print("===================\n")

print(response.text)