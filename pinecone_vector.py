import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ==================================================
# ENV
# ==================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ==================================================
# EMBEDDING MODEL
# ==================================================
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

DIMENSION = 384

# ==================================================
# PINECONE INIT
# ==================================================
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "onb"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# ==================================================
# LOAD PDF
# ==================================================
loader = PyPDFLoader("combined_ebola_pdf.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

texts = [c.page_content for c in chunks]

# ==================================================
# EMBEDDINGS
# ==================================================
vectors = embedding_model.encode(texts).tolist()

# ==================================================
# UPSERT
# ==================================================
data = [
    (str(i), vec, {"text": txt})
    for i, (vec, txt) in enumerate(zip(vectors, texts))
]

index.upsert(vectors=data)

print("✅ Ingestion complete")
