# embed_documents.py
import os
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from PyPDF2 import PdfReader
from openai import AzureOpenAI
import textwrap
 
# ===================== LOAD ENV ===================== #
load_dotenv()
 
# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBEDDING_MODEL_DEPLOYMENT = os.getenv("EMBEDDING_MODEL_DEPLOYMENT")
 
# CosmosDB
COSMOS_CONNECTION_STRING = os.getenv("COSMOS_CONNECTION_STRING")
DB_NAME = os.getenv("COSMOS_DATABASE_NAME")
COLLECTION_NAME = os.getenv("COSMOS_COLLECTION_NAME")
 
# PDF
PDF_FILE = Path(os.getenv("PDF_FILE"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
 
# ===================== INITIALIZE CLIENTS ===================== #
# Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)
 
# Cosmos DB client
mongo_client = MongoClient(COSMOS_CONNECTION_STRING)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]
print(" Connected to Cosmos DB successfully!\n")
 
# ===================== FUNCTIONS ===================== #
def extract_pdf_chunks(file_path, chunk_size=CHUNK_SIZE):
    """Extract text from PDF and split into chunks."""
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return textwrap.wrap(full_text, chunk_size)
 
def generate_embedding(text):
    """Generate embedding using Azure OpenAI."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding
 
def process_pdf(file_path):
    """Process PDF: extract chunks, generate embeddings, store in Cosmos DB."""
    print(f" Processing {file_path} ...")
    chunks = extract_pdf_chunks(file_path)
    print(f"✅ Extracted {len(chunks)} chunks.\n")
 
    for i, chunk in enumerate(chunks, start=1):
        embedding = generate_embedding(chunk)
        doc = {
            "chunk_id": i,
            "text": chunk,
            "embedding": embedding
        }
        collection.insert_one(doc)
        print(f" Stored chunk {i}/{len(chunks)}")
 
    print("\n All embeddings saved successfully in CosmosDB!")
 
# ===================== MAIN ===================== #
if __name__ == "__main__":
    if not PDF_FILE.exists():
        print(f"❌ File not found: {PDF_FILE}")
    else:
        process_pdf(PDF_FILE)