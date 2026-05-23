import os
import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv
from google import genai

# =========================
# ENV
# =========================
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("onb")

# =========================
# EMBED QUERY
# =========================
def embed_query(text):
    res = client.models.embed_content(
        model="text-embedding-004",
        contents=text,
        config={"task_type": "RETRIEVAL_QUERY"}
    )
    return res.embeddings[0].values

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

    model = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    return model.text

# =========================
# UI
# =========================
st.title("Ebola RAG Assistant")

q = st.chat_input("Ask question")

if q:
    st.write(generate_answer(q))
