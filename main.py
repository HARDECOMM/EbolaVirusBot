import os
import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv
import google.generativeai as genai

# ==================================================
# LOAD ENV VARIABLES
# ==================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY or PINECONE_API_KEY")

# ==================================================
# CONFIGURE GEMINI
# ==================================================
genai.configure(api_key=GEMINI_API_KEY)

# ==================================================
# PINECONE SETUP
# ==================================================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("onb")

# ==================================================
# EMBEDDING FUNCTION (FIXED)
# ==================================================
def embed_query(text: str):
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query"
    )
    return response["embedding"]

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

    matches = result.get("matches", [])

    contexts = []
    for match in matches:
        metadata = match.get("metadata", {})
        text = metadata.get("text", "")
        if text:
            contexts.append(text)

    return "\n\n".join(contexts)

# ==================================================
# GENERATE RESPONSE (GEMINI)
# ==================================================
def generate_response(user_query: str):
    context = retrieve_context(user_query)

    prompt = f"""
You are an Ebola Virus Guidance Assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say: "I don't know based on the provided documents."

Context:
{context}

Question:
{user_query}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text.strip()

# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(page_title="Ebola Virus Assistant", page_icon="🦠")
st.title("🦠 Ebola Virus Guidance Assistant")

# INIT CHAT HISTORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Hello! I'm your Ebola Virus Assistant. Ask me anything."
        }
    ]

# DISPLAY CHAT HISTORY
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# USER INPUT
user_input = st.chat_input("Ask your Ebola question...")

if user_input:
    # show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    # generate response
    with st.spinner("Thinking..."):
        response = generate_response(user_input)

    # show assistant response
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )
