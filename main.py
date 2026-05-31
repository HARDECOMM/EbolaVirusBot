import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from google import genai

# ==================================================
# ENV SETUP
# ==================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing API keys in environment variables")

# ==================================================
# GEMINI CLIENT (NEW SDK)
# ==================================================
client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "models/gemini-2.5-flash"

# ==================================================
# EMBEDDING MODEL (HUGGINGFACE)
# ==================================================
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ==================================================
# PINECONE SETUP
# ==================================================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("onb")

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

    matches = result.get("matches", [])
    if not matches:
        return ""

    return "\n\n".join(
        m["metadata"]["text"] for m in matches if "text" in m["metadata"]
    )

# ==================================================
# PROMPT ENGINE (FIXED RAG LOGIC)
# ==================================================
def generate_answer(query: str):
    context = retrieve_context(query)

    # detect weak retrieval
    has_context = len(context.strip()) > 50

    # =========================
    # STRICT RAG PROMPT
    # =========================
    if has_context:
        prompt = f"""
You are an Ebola Virus medical assistant.

IMPORTANT RULES:
- Answer ONLY using the context below
- Do NOT hallucinate or add extra medical facts
- If context is insufficient, say: "Not enough information in the knowledge base."

Context:
{context}

Question:
{query}

Answer:
"""
    else:
        prompt = f"""
You are an Ebola Virus medical assistant.

No relevant context was found in the knowledge base.

Answer ONLY if you are certain, otherwise say:
"I could not find this in the Ebola knowledge base."

Question:
{query}

Answer:
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(
    page_title="🦠 Ebola RAG Assistant",
    page_icon="🦠",
    layout="centered"
)

st.title("🦠 Ebola RAG Assistant")
st.write("Ask structured medical questions about Ebola Virus using your knowledge base.")

query = st.chat_input("Ask your question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(query)
            st.write(answer)

    # OPTIONAL DEBUG (UNCOMMENT IF NEEDED)
    # st.expander("Retrieved Context").write(retrieve_context(query))
