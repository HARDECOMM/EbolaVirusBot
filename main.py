import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# =========================
# ENV
# =========================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

# =========================
# EMBEDDING MODEL (NO API NEEDED)
# =========================
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# dimension = 384
DIMENSION = 384

# =========================
# PINECONE
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "onb"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# =========================
# LOAD PDF ON FIRST RUN
# =========================
@st.cache_resource
def load_and_index_data():
    loader = PyPDFLoader("combined_ebola_pdf.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    embeddings = embedding_model.encode(texts)

    vectors = []
    for i, (txt, emb) in enumerate(zip(texts, embeddings)):
        vectors.append((str(i), emb.tolist(), {"text": txt}))

    index.upsert(vectors=vectors)

    return "Indexed"

# run once per session
load_and_index_data()

# =========================
# EMBED QUERY
# =========================
def embed_query(text):
    return embedding_model.encode(text).tolist()

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

    return "\n\n".join([m["metadata"]["text"] for m in res["matches"]])

# =========================
# GENERATE ANSWER
# =========================
def generate_answer(query):
    context = retrieve_context(query)

    prompt = f"""
You are an Ebola Virus medical assistant.

Use ONLY the context below.

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

q = st.chat_input("Ask a question")

if q:
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(q)
            st.write(answer)
