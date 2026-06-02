from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# ==========================
# Load PDF
# ==========================

reader = PdfReader("data/sample.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

# ==========================
# Chunking
# ==========================

chunk_size = 300
chunk_overlap = 50

chunks = []

for i in range(
    0,
    len(text),
    chunk_size - chunk_overlap
):
    chunk = text[i:i + chunk_size]

    if chunk.strip():
        chunks.append(chunk)

print(f"Total Chunks: {len(chunks)}")

# ==========================
# Embeddings
# ==========================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(chunks)
## Convert to float32 for FAISS
## FAISS requires float32 data type for the embeddings
## This conversion is necessary to ensure compatibility with FAISS and to optimize memory usage.
## If we don't convert to float32, we might encounter errors or increased memory usage when working with FAISS.
embeddings = np.array(
    embeddings,
    dtype=np.float32
)

# ==========================
# Create FAISS Index
# ==========================
## The dimension of the embeddings is determined by the length of the embedding vectors produced by the SentenceTransformer model.
dimension = embeddings.shape[1]
## FAISS provides various index types for different use cases. IndexFlatL2 is a simple index that performs exact nearest neighbor search using L2 distance. It is suitable for small datasets and provides accurate results, but it may not be efficient for large datasets due to its linear search time complexity.
## In this example, we use IndexFlatL2 for simplicity, but for larger datasets, you might want to consider more advanced index types like IndexIVFFlat or IndexHNSW.
index = faiss.IndexFlatL2(dimension)
## Add the embeddings to the FAISS index. This step is crucial as it allows us to perform efficient similarity searches later on. The embeddings are added in batches, and the index will store them in a way that optimizes search performance.
index.add(embeddings)

print(
    f"Vectors Stored: {index.ntotal}"
)

# ==========================
# Save FAISS Index
# ==========================

faiss.write_index(
    index,
    "faiss_index.bin"
)

print(
    "FAISS index saved."
)

# ==========================
# Save Chunks
# ==========================
## We save the chunks in a separate file (chunks.pkl) using pickle. This allows us to retrieve the original text chunks later when we perform a search and want to display the relevant information to the user. By saving the chunks, we can easily access the corresponding text for any retrieved embedding from the FAISS index.
with open(
    "chunks.pkl",
    "wb"
) as f:
    ## The pickle.dump() function is used to serialize the chunks and save them to the file. This allows us to store the list of text chunks in a binary format that can be easily loaded back into memory when needed.
    pickle.dump(chunks, f)

print(
    "Chunks saved."
)