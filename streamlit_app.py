import io

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Page configuration
# =========================

st.set_page_config(
    page_title="AI Career Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================
# Sidebar
# =========================

st.sidebar.title("🤖 AI Career Assistant")

st.sidebar.markdown("""
### About this project

A RAG-based application that searches
relevant information from a CV.

### Technologies

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- Scikit-learn
- RAG

### Features

- 📄 PDF CV processing
- 🔎 Semantic search
- 📊 Relevance scoring
- 💬 CV question answering
""")


# =========================
# Main page
# =========================

st.title("🤖 AI Career Assistant")

st.markdown(
    "Upload your CV and ask questions about your skills, education, experience, or projects."
)

st.info(
    "Example: What are my technical skills?"
)


# =========================
# Example questions
# =========================

st.markdown("### 💡 Example Questions")

st.markdown("""
- What are my technical skills?
- What is my educational background?
- What work experience do I have?
- What projects have I worked on?
- What are my strengths?
""")


# =========================
# Upload CV
# =========================

uploaded_file = st.file_uploader(
    "📄 Upload your CV (PDF)",
    type=["pdf"]
)

if uploaded_file is None:
    st.warning("Please upload a PDF CV to start.")
    st.stop()


# =========================
# Read PDF
# =========================

try:
    pdf_bytes = uploaded_file.getvalue()

    reader = PdfReader(
        io.BytesIO(pdf_bytes)
    )

except Exception as error:
    st.error(f"Could not read the PDF: {error}")
    st.stop()


text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


if not text.strip():
    st.error(
        "No readable text was found in this PDF."
    )
    st.stop()


# =========================
# Create text chunks
# =========================

chunk_size = 500

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]

if not chunks:
    st.error("Could not create text chunks from the CV.")
    st.stop()


# =========================
# Load embedding model
# =========================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


with st.spinner("Loading AI model..."):
    model = load_model()


# =========================
# Create embeddings
# =========================

@st.cache_data(show_spinner=False)
def create_embeddings(chunk_list):
    return model.encode(chunk_list)


with st.spinner("Processing your CV..."):
    embeddings = create_embeddings(chunks)


st.success("CV loaded successfully.")


# =========================
# Ask question
# =========================

query = st.text_input(
    "🔎 Ask something about your CV:"
)


if query.strip():

    with st.spinner("Searching your CV..."):

        query_embedding = model.encode(
            [query]
        )

        similarities = cosine_similarity(
            query_embedding,
            embeddings
        )[0]

        best_index = similarities.argmax()

        score = float(
            similarities[best_index]
        )

    st.subheader(
        "📄 Relevant Information From Your CV"
    )

    if score < 0.15:

        st.warning(
            "I couldn't find enough relevant information in the CV. "
            "Try asking about skills, education, experience, or projects."
        )

    else:

        st.success(
            "Relevant information found in your CV."
        )

        st.write(
            chunks[best_index]
        )

    st.caption(
        f"Relevance score: {score:.3f}"
    )