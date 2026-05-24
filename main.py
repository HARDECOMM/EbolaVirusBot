import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ==================================================
# ENVIRONMENT VARIABLES
# ==================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# ==================================================
# HUGGINGFACE EMBEDDING MODEL (NO BILLING ISSUES)
# ==================================================
# This model converts text → vector (384 dimensions)
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DIM = 384

# ==================================================
# PINECONE SETUP
# ==================================================
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "onb"

# Create index automatically if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# ==================================================
# LOAD & INDEX PDF (RUNS ON FIRST APP START)
# ==================================================
@st.cache_resource
def build_vector_store():
    """
    Loads PDF, splits into chunks, embeds, and stores in Pinecone.
    Runs only once per Streamlit session.
    """

    loader = PyPDFLoader("combined_ebola_pdf.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]

    # Convert text chunks → embeddings
    embeddings = embedding_model.encode(texts)

    # Prepare vectors for Pinecone
    vectors = [
        (str(i), emb.tolist(), {"text": txt})
        for i, (txt, emb) in enumerate(zip(texts, embeddings))
    ]

    # Upload to Pinecone
    index.upsert(vectors=vectors)

    return "Vector store ready"


# Build vector store on startup
build_vector_store()

# ==================================================
# EMBED USER QUERY
# ==================================================
def embed_query(text: str):
    """Convert user question into vector"""
    return embedding_model.encode(text).tolist()

# ==================================================
# RETRIEVE CONTEXT FROM PINECONE
# ==================================================
def retrieve_context(query: str, top_k: int = 5):
    """Search Pinecone for similar documents"""

    vector = embed_query(query)

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    return "\n\n".join(
        match["metadata"]["text"]
        for match in result["matches"]
    )

# ==================================================
# SAFE GEMINI GENERATION (WITH FALLBACK MODEL)
# ==================================================
def generate_answer(query: str):
    """
    Generates answer using Gemini.
    Includes fallback model handling for Streamlit Cloud issues.
    """

    context = retrieve_context(query)

    prompt = f"""
You are an Ebola Virus medical assistant.

Use ONLY the context below to answer.
If not found, say you don't know.

Context:
{context}

Question:
{query}
"""

    # Try multiple models for stability (Streamlit-safe)
    models_to_try = [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro"
    ]

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text

        except Exception:
            # Try next model if current fails
            continue

    return "⚠️ Unable to generate response at this time. Please try again later."

# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(
    page_title="Ebola RAG Assistant",
    page_icon="🦠",
    layout="centered"
)

st.title("🦠 Ebola RAG Assistant")

st.markdown(
    "Ask medical questions about Ebola Virus using your knowledge base."
)

# User input
query = st.chat_input("Ask your question...")

if query:

    # Show user message
    with st.chat_message("user"):
        st.write(query)

    # Show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            answer = generate_answer(query)
            st.write(answer)
