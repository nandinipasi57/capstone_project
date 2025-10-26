from pathlib import Path
from PyPDF2 import PdfReader
from openai import AzureOpenAI
import os

class PDFProcessor:
    def __init__(self, azure_endpoint, azure_api_key, api_version, embedding_model):
        """
        Initialize PDF Processor for Azure OpenAI embeddings using SDK v1.0+
        """
        self.embedding_model = embedding_model
        self.client = AzureOpenAI(
            api_key=azure_api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )

    def process_pdf(self, pdf_path, chunk_size=1000, overlap=100):
        """
        Process a PDF file into chunks and generate embeddings
        """
        reader = PdfReader(pdf_path)
        documents = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue

            chunks = self._chunk_text(text, chunk_size, overlap)
            for idx, chunk in enumerate(chunks):
                embedding = self._get_embedding(chunk)
                if embedding:
                    documents.append({
                        "content": chunk,
                        "embedding": embedding,
                        "source": str(pdf_path),
                        "page": page_num + 1,
                        "chunk_index": idx
                    })

        return documents

    def _chunk_text(self, text, chunk_size, overlap):
        """
        Split text into overlapping chunks
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _get_embedding(self, text):
        """
        Generate embedding using Azure OpenAI (v1.0+ SDK)
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,  # Azure deployment name
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return []
