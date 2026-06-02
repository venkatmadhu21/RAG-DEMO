from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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

# Embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector Store

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# Retriever

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Query

question = input(
    "\nAsk Question: "
)

results = retriever.invoke(
    question
)

print("\nRetrieved Documents:\n")

for i, doc in enumerate(results):
    print(f"\nChunk {i+1}")
    print("-" * 40)
    print(doc.page_content)