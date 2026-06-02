from pypdf import PdfReader

pdf_path = "data/sample.pdf"

reader = PdfReader(pdf_path)

print(f"Total Pages: {len(reader.pages)}")

text = ""

for page in reader.pages:
    text += page.extract_text()

print("\nFirst 1000 Characters:\n")
print(text[:1000])