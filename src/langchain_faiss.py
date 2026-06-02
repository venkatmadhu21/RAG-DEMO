from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==========================
# Load PDF
# ==========================

loader = PyPDFLoader(
    "data/sample.pdf"
)

documents = loader.load()

# ==========================
# Split Documents
# ==========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(
    documents
)

print(f"Chunks Created: {len(chunks)}")

# ==========================
# Embeddings
# ==========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================
# Vector Store
# ==========================

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS Vector Store Created")
print(type(vectorstore))