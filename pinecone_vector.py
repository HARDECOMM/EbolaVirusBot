import os
import time
import re
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

# ==================================================
# ENV SETUP
# ==================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY or PINECONE_API_KEY")

# ==================================================
# GEMINI CONFIG
# ==================================================
genai.configure(api_key=GEMINI_API_KEY)

# ==================================================
# TEXT SPLITTING
# ==================================================
def split_docs(documents, chunk_size=1500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# ==================================================
# RETRY HELPER
# ==================================================
def parse_retry_wait_time(error):
    match = re.search(r"(\d+)s", str(error))
    return int(match.group(1)) if match else 20

# ==================================================
# EMBEDDING (DOCUMENT)
# ==================================================
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
                print(f"Embedding error (attempt {attempt+1}): {e}")

                wait_time = parse_retry_wait_time(e)
                time.sleep(wait_time)

                if attempt == 2:
                    raise

    return embeddings

# ==================================================
# PINECONE SETUP
# ==================================================
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "onb"

existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Created Pinecone index: {index_name}")
    time.sleep(60)

index = pc.Index(index_name)

# ==================================================
# LOAD PDF
# ==================================================
pdf_file_path = "combined_ebola_pdf.pdf"
loader = PyPDFLoader(pdf_file_path)
documents = loader.load()

docs = split_docs(documents)

texts = [doc.page_content for doc in docs]

# ==================================================
# EMBED DOCUMENTS
# ==================================================
print("\n🚀 Embedding documents...\n")
embeddings = embed_documents(texts)

# ==================================================
# PREPARE VECTORS
# ==================================================
vectors = [
    (str(i), emb, {"text": text})
    for i, (emb, text) in enumerate(zip(embeddings, texts))
]

# ==================================================
# UPSERT TO PINECONE
# ==================================================
def batch_upsert(index, vectors, batch_size=100):
    batches = [
        vectors[i:i + batch_size]
        for i in range(0, len(vectors), batch_size)
    ]

    for i, batch in enumerate(tqdm(batches, desc="Upserting")):
        for attempt in range(3):
            try:
                index.upsert(vectors=batch)
                break
            except Exception as e:
                print(f"Upsert error batch {i+1}: {e}")
                time.sleep(5 * (attempt + 1))
                if attempt == 2:
                    raise

# ==================================================
# RUN PIPELINE
# ==================================================
print("\n🚀 Starting Pinecone upsert...\n")
batch_upsert(index, vectors)
print("\n✅ Ingestion complete!\n")
