import os
import streamlit as st
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("onb")

# =========================
# EMBED QUERY (FIXED)
# =========================
def embed_query(text):
    res = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query"
    )
    return res["embedding"]

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
# GENERATE
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
# UI
# =========================
st.title("Ebola RAG Assistant")

q = st.chat_input("Ask a question")

if q:
    st.write(generate_answer(q))
