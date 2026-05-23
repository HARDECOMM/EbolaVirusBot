import os
import time
import re
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

# =========================
# ENV
# =========================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing API keys")

genai.configure(api_key=GEMINI_API_KEY)

# =========================
# SPLIT DOCS
# =========================
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

# =========================
# EMBEDDING (FIXED)
# =========================
def embed_documents(texts):
    embeddings = []

    for text in texts:
        for attempt in range(3):
            try:
                res = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )

                embeddings.append(res["embedding"])
                break

            except Exception as e:
                wait = 10
                print(f"Retrying embedding: {e}")
                time.sleep(wait)

                if attempt == 2:
                    raise

    return embeddings

# =========================
# PINECONE
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "onb"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# =========================
# LOAD PDF
# =========================
loader = PyPDFLoader("combined_ebola_pdf.pdf")
docs = split_docs(loader.load())

texts = [d.page_content for d in docs]

# =========================
# EMBED + UPLOAD
# =========================
embeddings = embed_documents(texts)

vectors = [
    (str(i), emb, {"text": txt})
    for i, (emb, txt) in enumerate(zip(embeddings, texts))
]

def upsert(vectors, batch_size=100):
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])

upsert(vectors)

print("DONE INGESTION")
