import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now proceed with the rest of your imports
from database.cosmos_vector_db import CosmosVectorDB  # Import the CosmosVectorDB class
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT_NAME = os.getenv("EMBEDDING_MODEL_DEPLOYMENT")

# CosmosDB
COSMOS_CONNECTION_STRING = os.getenv("COSMOS_CONNECTION_STRING")
DB_NAME = os.getenv("COSMOS_DATABASE_NAME")
COLLECTION_NAME = os.getenv("COSMOS_COLLECTION_NAME")

class Retrieval:
    def __init__(self):
        # Initialize Azure OpenAI client
        try:
            self.client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_KEY,
                api_version=AZURE_OPENAI_API_VERSION
            )
            print("Azure OpenAI client initialized.")
        except Exception as e:
            print(f"❌ Failed to initialize Azure OpenAI client: {e}")
            exit(1)

        # Initialize Cosmos DB using CosmosVectorDB class
        try:
            cosmos_db = CosmosVectorDB(COSMOS_CONNECTION_STRING, DB_NAME, COLLECTION_NAME)
            self.mongo_client = cosmos_db.collection  # Get the collection from CosmosVectorDB
            print(" Connected to Cosmos DB.")
        except Exception as e:
            print(f"❌ MongoDB connection error: {e}")
            exit(1)

    def generate_query_embedding(self, query):
        """Generate embedding for a query string using Azure OpenAI."""
        response = self.client.embeddings.create(
            model=AZURE_DEPLOYMENT_NAME,
            input=query
        )
        return response.data[0].embedding

    def search_similar_chunks(self, query, top_k=5):
        """Search top_k similar chunks from CosmosDB using vector similarity."""
        query_embedding = self.generate_query_embedding(query)

        # MongoDB vector search aggregation pipeline
        pipeline = [
            {
                "$search": {
                    "cosmosSearch": {
                        "vector": query_embedding,
                        "path": "embedding",
                        "k": top_k
                    },
                    "returnStoredSource": True
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "text": 1,
                    "embedding": 1,
                    "similarity_score": {"$meta": "searchScore"}
                }
            }
        ]

        results = list(self.mongo_client.aggregate(pipeline))
        return results

    print("Retrieval module initialized successfully.")
 