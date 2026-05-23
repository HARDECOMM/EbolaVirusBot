import os
import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv

import vertexai
from vertexai.language_models import TextEmbeddingModel

import google.generativeai as genai

# =========================
# ENV
# =========================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

# =========================
# INIT
# =========================
genai.configure(api_key=GEMINI_API_KEY)

vertexai.init(project=PROJECT_ID, location="us-central1")

embedding_model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@003"
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("onb")

# =========================
# EMBED QUERY
# =========================
def embed_query(text):
    return embedding_model.get_embeddings([text])[0].values

# =========================
# RETRIEVE
# =========================
def retrieve_context(query):
    vector = embed_query(query)

    res = index.query(
        vector=vector,
        top_k=5,
        include_metadata=True
    )

    return "\n\n".join(
        [m["metadata"]["text"] for m in res["matches"]]
    )

# =========================
# GENERATE ANSWER
# =========================
def generate_answer(query):
    context = retrieve_context(query)

    prompt = f"""
You are an Ebola medical assistant.

Context:
{context}

Question:
{query}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.generate_content(prompt).text

# =========================
# STREAMLIT UI
# =========================
st.title("🦠 Ebola RAG Assistant")

q = st.chat_input("Ask your question")

if q:
    st.write(generate_answer(q))
