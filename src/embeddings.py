# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")

# sentences = [
#     "I love dogs",
#     "Dogs are my favorite animals",
#     "I enjoy playing cricket"
# ]

# embeddings = model.encode(sentences)

# print("Embedding Shape:", embeddings.shape)
# print("\nFirst 10 values of first embedding:\n")
# print(embeddings[0][:10])

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("all-MiniLM-L6-v2")

s1 = "I love dogs"
s2 = "Dogs are my favorite animals"
s3 = "I enjoy playing cricket"

e1 = model.encode(s1)
e2 = model.encode(s2)
e3 = model.encode(s3)
s4 = "Puppies are adorable"
e4 = model.encode(s4)

print("Dog vs Dog:")
print(cos_sim(e1, e2))

print("\nDog vs Cricket:")
print(cos_sim(e1, e3))

print("\nDog vs Puppies:")
print(cos_sim(e1, e4))