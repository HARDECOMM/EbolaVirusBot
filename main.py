import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from google import genai

# ==================================================
# ENV
# ==================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==================================================
# GEMINI CLIENT (WORKING SDK)
# ==================================================
client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "models/gemini-2.5-flash"

# ==================================================
# EMBEDDINGS (HUGGINGFACE - STABLE)
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

index = pc.Index(index_name)

# ==================================================
# EMBED QUERY
# ==================================================
def embed_query(text: str):
    return embedding_model.encode(text).tolist()

# ==================================================
# RETRIEVE CONTEXT
# ==================================================
def retrieve_context(query: str, top_k: int = 3):
    vector = embed_query(query)

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    texts = [
        match["metadata"]["text"]
        for match in result["matches"]
    ]

    return "\n\n".join(texts)[:3000]

# ==================================================
# GEMINI CALL (STABLE)
# ==================================================
def call_gemini(prompt: str):
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text

    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

# ==================================================
# RAG RESPONSE
# ==================================================
def generate_answer(query: str):
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

    return call_gemini(prompt)

# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(
    page_title="Ebola RAG Assistant",
    page_icon="🦠"
)

st.title("🦠 Ebola RAG Assistant")

query = st.chat_input("Ask your question about Ebola...")

if query:

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(query)
            st.write(answer)
