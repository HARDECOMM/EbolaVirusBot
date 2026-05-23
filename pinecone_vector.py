import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from google import genai

# =========================
# ENV
# =========================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

# =========================
# SPLIT DOCS
# =========================
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

# =========================
# EMBEDDING (DOCUMENT)
# =========================
def embed_documents(texts):
    embeddings = []

    for text in texts:
        res = client.models.embed_content(
            model="embedding-001",
            contents=text,
            config={"task_type": "RETRIEVAL_DOCUMENT"}
        )
        embeddings.append(res.embeddings[0].values)

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

index.upsert(vectors=vectors)

print("INGESTION COMPLETE")
