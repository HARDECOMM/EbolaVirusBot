import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ==================================================
# ENVIRONMENT SETUP
# ==================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# ==================================================
# EMBEDDING MODEL (HUGGINGFACE - STABLE)
# ==================================================
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

DIMENSION = 384

# ==================================================
# PINECONE SETUP
# ==================================================
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "onb"

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# ==================================================
# EMBEDDING FUNCTION
# ==================================================
def embed_query(text: str):
    """Convert query into vector using HuggingFace model"""
    return embedding_model.encode(text).tolist()

# ==================================================
# RETRIEVE CONTEXT FROM PINECONE
# ==================================================
def retrieve_context(query: str, top_k: int = 3):
    """Get relevant chunks from Pinecone"""

    vector = embed_query(query)

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    texts = [match["metadata"]["text"] for match in result["matches"]]

    # prevent token overflow
    context = "\n\n".join(texts)

    return context[:2500]

# ==================================================
# GENERATE ANSWER (FIXED GEMINI)
# ==================================================
def generate_answer(query: str):
    """Generate response using Gemini (stable model)"""

    context = retrieve_context(query)

    prompt = f"""
You are an Ebola Virus medical assistant.

Use ONLY the context below.
If answer is not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

    try:
        # ✅ FIXED MODEL (WORKING FOR YOUR API)
        model = genai.GenerativeModel("gemini-pro")

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(
    page_title="Ebola RAG Assistant",
    page_icon="🦠",
    layout="centered"
)

st.title("🦠 Ebola RAG Assistant")
st.write("Ask questions using your Ebola knowledge base.")

query = st.chat_input("Ask your question...")

if query:

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(query)
            st.write(answer)
