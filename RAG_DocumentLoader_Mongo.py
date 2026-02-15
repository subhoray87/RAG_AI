import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import uuid
import re
from datetime import datetime
from dotenv import load_dotenv

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# MongoDB
from pymongo import MongoClient

# =========================
# CONFIG
# =========================
load_dotenv()

PDF_PATH = r".\Docs\kafka.pdf"

# =========================
# LOAD PDF
# =========================
loader = PyPDFLoader(PDF_PATH)
pdf_documents = loader.load()

print(f"Loaded {len(pdf_documents)} pages")

# =========================
# METADATA
# =========================
document_id = str(uuid.uuid4())
ingestion_time = datetime.now().isoformat()

# =========================
# CLEANING FUNCTION
# =========================
def clean_pdf_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"FT\d+[_\s]?\d+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# Apply cleaning + metadata enrichment
for doc in pdf_documents:
    doc.page_content = clean_pdf_text(doc.page_content)

    doc.metadata.update({
        "document_id": document_id,
        "document_name": os.path.basename(PDF_PATH),
        "document_type": "Rhapsody PDF",
        "ingested_at": ingestion_time
    })

# =========================
# SPLIT INTO CHUNKS
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80
)

chunks = splitter.split_documents(pdf_documents)
print(f"Split into {len(chunks)} chunks")

# =========================
# LOAD EMBEDDING MODEL
# =========================
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

# =========================
# GENERATE EMBEDDINGS
# =========================
texts = [doc.page_content for doc in chunks]
vectors = embeddings_model.embed_documents(texts)

print("Embeddings generated")

# =========================
# CONNECT TO MONGO
# =========================
client=MongoClient(
    "mongodb+srv://subhoray87_db_user:owUuRhttrBf3Po3x@rag.rtev49n.mongodb.net/?appName=RAG",
    tls=True,
    tlsAllowInvalidCertificates=True)
collection=client["sample_mflix"]["RHpdf"]

# =========================
# BUILD RECORDS
# =========================
records = []

for doc, vector in zip(chunks, vectors):
    records.append({
        "_id": str(uuid.uuid4()),
        "text": doc.page_content,
        "embedding": vector,
        "metadata": doc.metadata
    })

# =========================
# INSERT INTO MONGO
# =========================
BATCH_SIZE = 100

for i in range(0, len(records), BATCH_SIZE):
    batch = records[i:i+BATCH_SIZE]
    collection.insert_many(batch, ordered=False)

print(f"Inserted {len(records)} chunks into MongoDB")
print("Ingestion complete ✅")