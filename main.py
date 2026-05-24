import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from google import genai

# ==================================================
# ENV SETUP
# ==================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==================================================
# GEMINI CLIENT (NEW SDK)
# ==================================================
client = genai.Client(api_key=GEMINI_API_KEY)

# ==================================================
# EMBEDDINGS (HUGGINGFACE)
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
# EMBEDDING FUNCTION
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

    texts = [m["metadata"]["text"] for m in result["matches"]]

    return "\n\n".join(texts)[:2500]  # prevent token overflow

# ==================================================
# GEMINI MODEL FALLBACK SYSTEM (FIXES 404)
# ==================================================
def call_gemini(prompt: str):
    """
    Try multiple models because your API access is restricted.
    This prevents 404 crashes completely.
    """

    models_to_try = [
        "gemini-1.0-pro",
        "gemini-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ]

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if response and response.text:
                return response.text

        except Exception:
            continue  # try next model

    return "❌ All Gemini models failed for this API key."

# ==================================================
# GENERATE ANSWER
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
