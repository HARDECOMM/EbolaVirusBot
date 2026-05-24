import os
import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ==================================================
# ENV SETUP
# ==================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==================================================
# GEMINI CONFIG
# ==================================================
genai.configure(api_key=GEMINI_API_KEY)

# ==================================================
# HUGGINGFACE EMBEDDING MODEL
# all-MiniLM-L6-v2 = 384 dimensions
# ==================================================
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================================
# PINECONE SETUP
# ==================================================
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("onb")

# ==================================================
# EMBED QUERY
# ==================================================
def embed_query(text):
    embedding = embedding_model.encode(text)
    return embedding.tolist()

# ==================================================
# RETRIEVE CONTEXT
# ==================================================
def retrieve_context(query, top_k=5):
    vector = embed_query(query)

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    contexts = []

    for match in result["matches"]:
        contexts.append(match["metadata"]["text"])

    return "\n\n".join(contexts)

# ==================================================
# GENERATE ANSWER
# ==================================================
def generate_answer(query):
    context = retrieve_context(query)

    prompt = f"""
You are an Ebola Virus medical assistant.

Use ONLY the provided context to answer the question.

If the answer is not in the context, say:
"I could not find that information in the knowledge base."

Context:
{context}

Question:
{query}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(prompt)

    return response.text

# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(
    page_title="Ebola RAG Assistant",
    page_icon="🦠",
    layout="centered"
)

st.title("🦠 Ebola Virus RAG Assistant")

st.markdown(
    "Ask questions about Ebola Virus using the uploaded medical documents."
)

# ==================================================
# CHAT INPUT
# ==================================================
user_question = st.chat_input(
    "Ask a question about Ebola Virus..."
)

if user_question:

    # User message
    with st.chat_message("user"):
        st.markdown(user_question)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Generating answer..."):

            answer = generate_answer(user_question)

            st.markdown(answer)
