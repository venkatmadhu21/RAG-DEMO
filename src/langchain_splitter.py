from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================
# Load PDF
# ==========================

loader = PyPDFLoader(
    "data/sample.pdf"
)

documents = loader.load()

print(
    f"Pages Loaded: {len(documents)}"
)

# ==========================
# Split Documents
# ==========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(
    documents
)

print(
    f"Total Chunks: {len(chunks)}"
)

print("\nFirst Chunk:\n")

print(
    chunks[0].page_content
)