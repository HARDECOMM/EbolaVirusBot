import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from google import genai

# ==================================================
# ENV SETUP
# ==================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==================================================
# GEMINI CLIENT
# ==================================================
client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "models/gemini-2.5-flash"

# ==================================================
# EMBEDDING MODEL
# ==================================================
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

DIMENSION = 384

# ==================================================
# PINECONE SETUP
# ==================================================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("onb")

# ==================================================
# EMBED QUERY
# ==================================================
def embed_query(text: str):
    return embedding_model.encode(text).tolist()

# ==================================================
# RETRIEVE CONTEXT (IMPROVED)
# ==================================================
def retrieve_context(query: str, top_k: int = 4):
    vector = embed_query(query)

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    if not result.get("matches"):
        return ""

    texts = [m["metadata"]["text"] for m in result["matches"]]

    return "\n\n".join(texts)

# ==================================================
# HYBRID GEMINI RESPONSE (IMPORTANT FIX)
# ==================================================
def generate_answer(query: str):
    context = retrieve_context(query)

    # detect weak retrieval
    use_fallback = len(context.strip()) < 80

    prompt = f"""
You are an expert Ebola Virus medical assistant.

You MUST respond in a structured medical format:

### 🦠 Overview
### 📊 Key Facts
### ⚠️ Symptoms
### 🔬 Transmission
### 🧪 Diagnosis
### 💊 Treatment
### 🛡️ Prevention

Rules:
- If context is available, prioritize it.
- If context is weak, use general medical knowledge about Ebola.
- Never go outside Ebola-related medical information.
- Be clear, accurate, and educational.

Context:
{context if context else "No context available from knowledge base."}

Question:
{query}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

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
st.write("Ask structured medical questions about Ebola Virus.")

query = st.chat_input("Ask your question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(query)
            st.write(answer)
