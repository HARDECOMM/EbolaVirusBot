import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import vertexai
from vertexai.language_models import TextEmbeddingModel

# =========================
# ENV
# =========================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

# =========================
# INIT VERTEX AI
# =========================
vertexai.init(project=PROJECT_ID, location="us-central1")

embedding_model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@003"
)

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
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)
texts = [c.page_content for c in chunks]

# =========================
# EMBEDDINGS (VERTEX AI)
# =========================
def embed_texts(text_list):
    embeddings = embedding_model.get_embeddings(text_list)
    return [e.values for e in embeddings]

print("Embedding documents...")
vectors = embed_texts(texts)

# =========================
# UPSERT
# =========================
data = [
    (str(i), vec, {"text": txt})
    for i, (vec, txt) in enumerate(zip(vectors, texts))
]

index.upsert(vectors=data)

print("INGESTION COMPLETE")
