import os
import time
import re
import concurrent.futures
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv
from google.generativeai import client as genai_client

# ==================================================
# ENV SETUP
# ==================================================
load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# ==================================================
# GEMINI CLIENT (v1)
# ==================================================
genai = genai_client.Client(api_key=GEMINI_API_KEY)

# ==================================================
# HELPERS
# ==================================================
def split_docs(documents, chunk_size=1500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def parse_retry_wait_time(error):
    if hasattr(error, "response") and error.response is not None:
        retry_after = error.response.headers.get("Retry-After")
        if retry_after:
            return int(retry_after)
    match = re.search(r"(\d+)s", str(error))
    return int(match.group(1)) if match else 20

def embed_batch_with_retry(batch_contents, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            embeddings = []
            for text in batch_contents:
                res = genai.embeddings.create(
                    model="text-embedding-004",
                    content=text
                )
                embeddings.append(res.embedding.values)
            return embeddings
        except Exception as e:
            print(f"Embedding error (attempt {attempt+1}): {e}")
            wait_time = parse_retry_wait_time(e)
            print(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)
            if attempt == max_attempts - 1:
                raise

def concurrent_embed_documents(documents, batch_size=50, max_workers=4):
    all_embeddings = []
    all_contents = []
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_contents = [doc.page_content for doc in batch]
            futures.append(
                (executor.submit(embed_batch_with_retry, batch_contents), batch_contents)
            )

        for future, contents in tqdm(futures, desc="Embedding batches", total=len(futures)):
            batch_embeddings = future.result()
            all_embeddings.extend(batch_embeddings)
            all_contents.extend(contents)

    return all_embeddings, all_contents

# ==================================================
# PINECONE SETUP
# ==================================================
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "onb"

existing_indexes = pc.list_indexes()
existing_index_names = [idx.name for idx in existing_indexes.indexes]

if index_name not in existing_index_names:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Created Pinecone index: {index_name}")
    time.sleep(60)

pinecone_index = pc.Index(index_name)

# ==================================================
# LOAD & PROCESS DOCUMENTS
# ==================================================
pdf_file_path = "combined_ebola_pdf.pdf"
loader = PyPDFLoader(pdf_file_path)
documents = loader.load()

docs = split_docs(documents)

# ==================================================
# EMBED & UPSERT
# ==================================================
all_embeddings, all_contents = concurrent_embed_documents(
    docs,
    batch_size=50,
    max_workers=4
)

vectors = [
    (str(i), embedding, {"text": content})
    for i, (embedding, content) in enumerate(zip(all_embeddings, all_contents))
]

def batch_upsert(index, vectors, batch_size=100):
    batches = [vectors[i:i + batch_size] for i in range(0, len(vectors), batch_size)]
    for batch_num, batch in enumerate(tqdm(batches, desc="Upserting batches")):
        for attempt in range(3):
            try:
                index.upsert(vectors=batch)
                break
            except Exception as e:
                print(f"Upsert error (batch {batch_num+1}, attempt {attempt+1}): {e}")
                time.sleep(10 * (attempt + 1))
                if attempt == 2:
                    raise

print("\n🚀 Starting Pinecone upserts...\n")
batch_upsert(pinecone_index, vectors)
print("\n✅ Pinecone vector storage complete!\n")
