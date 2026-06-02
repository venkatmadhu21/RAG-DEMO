from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load PDF

loader = PyPDFLoader(
    "data/sample.pdf"
)

documents = loader.load()

# Split

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(
    documents
)

print(f"Total Chunks: {len(chunks)}")

# Embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embeddings.embed_query(
    chunks[0].page_content
)

print("\nEmbedding Dimension:")
print(len(vector))

print("\nFirst 10 Values:")
print(vector[:10])