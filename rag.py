from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Load PDF
pdf_folder = Path("documents")
pdf_files = list(pdf_folder.glob("*.pdf"))

if not pdf_files:
    print("No PDF found.")
    exit()

pdf_path = pdf_files[0]
reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


# Create chunks
chunk_size = 500

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]

print("===== PDF LOADED =====")
print("File:", pdf_path.name)
print("Chunks:", len(chunks))


# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("\n===== EMBEDDINGS CREATED =====")
print("Number of embeddings:", len(embeddings))
print("Embedding size:", len(embeddings[0]))
from sklearn.metrics.pairwise import cosine_similarity

while True:
    query = input("\nAsk about the CV (or type 'exit'): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    best_index = similarities.argmax()
    score = similarities[best_index]

    print("\n===== AI CAREER ASSISTANT =====")

    if score < 0.15:
        print("I couldn't find enough relevant information in the CV.")
    else:
        print("Based on the CV:")
        print(chunks[best_index])

    print("\nRelevance Score:", round(score, 3))