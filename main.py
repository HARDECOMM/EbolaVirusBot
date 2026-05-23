import os
import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv
import google.generativeai as genai

# ==================================================
# Load environment variables (local dev only)
# Streamlit Cloud uses Secrets automatically
# ==================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("GEMINI_API_KEY or PINECONE_API_KEY is missing! Check your .env or Streamlit Secrets.")

# ==================================================
# Configure Gemini SDK
# ==================================================
genai.configure(api_key=GEMINI_API_KEY)

# ==================================================
# Pinecone setup
# ==================================================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("onb")  # Ensure the same index as your ingestion script

# ==================================================
# Embed query
# ==================================================
def embed_query(text):
    response = genai.embed_content(
        model="text-embedding-004",
        content=text,
        task_type="retrieval_query"
    )

    return response["embedding"]"]

# ==================================================
# Retrieve context from Pinecone
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
# Generate response from Gemini chat
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

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text.strip()

# ==================================================
# Streamlit UI
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
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Generate assistant response
    with st.spinner("Thinking..."):
        response = generate_response(user_input)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})
