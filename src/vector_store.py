# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# # Load model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Documents
# documents = [
#     "I love dogs",
#     "Dogs are my favorite animals",
#     "I enjoy playing cricket",
#     "Puppies are adorable"
# ]

# # Generate embeddings
# embeddings = model.encode(documents)

# # Convert to float32 (required by FAISS)
# embeddings = np.array(embeddings).astype("float32")

# # Create FAISS index
# dimension = embeddings.shape[1]

# index = faiss.IndexFlatL2(dimension)

# # Add vectors
# index.add(embeddings)

# print("Total vectors stored:", index.ntotal)


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Documents
documents = [
    "I love dogs",
    "Dogs are my favorite animals",
    "I enjoy playing cricket",
    "Puppies are adorable"
]

# Create embeddings
embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype("float32")

# Create index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Store vectors
index.add(embeddings)

print("Total vectors stored:", index.ntotal)

# -------------------------
# QUERY
# -------------------------

query = "I like dogs"
k =3
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

distances, indices = index.search(query_embedding, k)

print("\nDistances:")
print(distances)

print("\nIndices:")
print(indices)

print("\nNearest Documents:")

for i in indices[0]:
    print(documents[i])