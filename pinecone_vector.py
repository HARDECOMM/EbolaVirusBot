import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ==================================================
# ENV SETUP
# ==================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ==================================================
# HUGGINGFACE EMBEDDING MODEL
# all-MiniLM-L6-v2 = 384 dimensions
# ==================================================
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================================
# PINECONE SETUP
# ==================================================
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "onb"

existing_indexes = pc.list_indexes().names()

# ==================================================
# CREATE INDEX IF NOT EXISTS
# IMPORTANT:
# dimension MUST match embedding size
# MiniLM = 384 dimensions
# ==================================================
if index_name not in existing_indexes:

    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# ==================================================
# LOAD PDF DOCUMENT
# ==================================================
loader = PyPDFLoader("combined_ebola_pdf.pdf")

documents = loader.load()

# ==================================================
# SPLIT DOCUMENTS
# ==================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

texts = [chunk.page_content for chunk in chunks]

# ==================================================
# EMBED DOCUMENTS
# ==================================================
print("Generating embeddings...")

embeddings = embedding_model.encode(texts)

# ==================================================
# PREPARE VECTORS
# ==================================================
vectors = []

for i, (text, embedding) in enumerate(zip(texts, embeddings)):

    vectors.append(
        (
            str(i),
            embedding.tolist(),
            {"text": text}
        )
    )

# ==================================================
# UPSERT TO PINECONE
# ==================================================
print("Uploading vectors to Pinecone...")

index.upsert(vectors=vectors)

print("✅ Pinecone upload complete!")
