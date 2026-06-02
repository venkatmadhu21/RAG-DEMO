from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "data/sample.pdf"
)

documents = loader.load()

print(f"Pages Loaded: {len(documents)}")

print("\nFirst Page:\n")

print(documents[0].page_content[:500])