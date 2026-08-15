import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Career Assistant",
    page_icon="🤖"
)
st.sidebar.title("🤖 AI Career Assistant")

st.sidebar.markdown("""
### About this project

This application uses RAG to answer questions from a CV.

### Technologies
- Python
- Streamlit
- Sentence Transformers
- NumPy
- PyPDF
- RAG

### Features
- 📄 PDF CV processing
- 🔎 Semantic search
- 📊 Relevance scoring
- 💬 CV question answering
""")
st.title("🤖 AI Career Assistant")
st.markdown(
    "Ask questions about your CV and find relevant information instantly."
)

st.info(
    "Try questions like: "
    "What are my technical skills? "
    "What is my educational background? "
    "What work experience do I have?"
)
st.markdown("### 💡 Example Questions")

st.markdown("""
- What are my technical skills?
- What is my educational background?
- What work experience do I have?
- What projects have I worked on?
- What are my strengths?
""")

# -------------------------
# Load CV
# -------------------------

pdf_folder = Path("documents")
pdf_files = list(pdf_folder.glob("*.pdf"))

if not pdf_files:
    st.error("No PDF found in the documents folder.")
    st.stop()

pdf_path = pdf_files[0]

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


# -------------------------
# Create CV chunks
# -------------------------

chunk_size = 500

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]


# -------------------------
# Load embedding model
# -------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# -------------------------
# Create embeddings
# -------------------------

@st.cache_data
def create_embeddings(_chunks):
    return model.encode(_chunks)


embeddings = create_embeddings(chunks)


# -------------------------
# Ask question
# -------------------------

query = st.text_input(
    "Ask something about your CV:"
)


if query:

    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    best_index = similarities.argmax()

    score = similarities[best_index]

    st.subheader("📄 Relevant Information From Your CV")

if score < 0.15:
    st.warning(
        "I couldn't find enough relevant information in the CV. "
        "Try asking about skills, education, experience, or projects."
    )
else:
    st.success("Relevant information found in your CV.")
    st.write(chunks[best_index])

st.caption(
    f"Relevance score: {score:.3f}"
)