import os
import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv
from google.generativeai import client as genai_client

# ==================================================
# ENV SETUP
# ==================================================
load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# ==================================================
# CLIENTS
# ==================================================
genai = genai_client.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("onb")  # same index name as ingestion

# ==================================================
# EMBEDDING FUNCTION (Gemini v1)
# ==================================================
def embed_query(text: str):
    res = genai.embeddings.create(
        model="text-embedding-004",
        content=text
    )
    return res.embedding.values

# ==================================================
# RETRIEVE CONTEXT FROM PINECONE
# ==================================================
def retrieve_context(query: str, top_k: int = 5):
    query_embedding = embed_query(query)

    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    contexts = []
    for match in result["matches"]:
        contexts.append(match["metadata"]["text"])

    return "\n\n".join(contexts)

# ==================================================
# GENERATE RESPONSE (Gemini v1 Chat)
# ==================================================
def generate_response(user_query: str):
    context = retrieve_context(user_query)

    prompt = f"""
You are an Ebola Virus Guidance Assistant.
Use the context below to answer the user's question accurately.
If the answer is not in the context, say you don't know.

Context:
{context}

User question:
{user_query}
"""

    response = genai.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    return response.text.strip()

# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(page_title="Ebola Virus Assistant", page_icon="🦠")

st.title("🦠 Ebola Virus Guidance Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Hello! I'm your Ebola Virus Guidance Assistant. How can I assist you today?"
        }
    ]

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
user_input = st.chat_input("Ask your Ebola Virus question:")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    with st.spinner("Thinking..."):
        response = generate_response(user_input)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )
