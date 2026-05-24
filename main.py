import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ==================================================
# ENV SETUP
# ==================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
# EMBED QUERY
# ==================================================
def embed_query(text):
    return embedding_model.encode(text).tolist()

# ==================================================
# RETRIEVE CONTEXT (SAFE + LIMITED)
# ==================================================
def retrieve_context(query, top_k=3):
    vector = embed_query(query)

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    texts = [m["metadata"]["text"] for m in result["matches"]]

    # IMPORTANT: prevent token overflow
    context = "\n\n".join(texts)

    # HARD LIMIT (prevents Gemini crash)
    return context[:2500]

# ==================================================
# GENERATE ANSWER (FIXED + DEBUG SAFE)
# ==================================================
def generate_answer(query):
    context = retrieve_context(query)

    prompt = f"""
You are an Ebola Virus medical assistant.

Use ONLY the context below.
If the answer is not in context, say "I don't know".

Context:
{context}

Question:
{query}
"""

    try:
        # ONLY USE ONE STABLE MODEL
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(prompt)

        # SAFE RETURN
        if response and hasattr(response, "text"):
            return response.text

        return "⚠️ Empty response from Gemini."

    except Exception as e:
        # SHOW REAL ERROR (NO MORE SILENT FAIL)
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

st.markdown("Ask questions about Ebola Virus using your knowledge base.")

query = st.chat_input("Ask your question...")

if query:

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(query)
            st.write(answer)
