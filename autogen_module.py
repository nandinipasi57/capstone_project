# autogen_module.py
import os
from dotenv import load_dotenv
from retrieve_documents import Retrieval
from openai import AzureOpenAI

# Load .env file
load_dotenv()

# ===================== CONFIG ===================== #
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AUTOGEN_DEPLOYMENT = os.getenv("AUTOGEN_DEPLOYMENT")  # chat model deployment

# ===================== INITIALIZE MODULES ===================== #

# Initialize Retrieval module (fetches from Cosmos DB)
retriever = Retrieval()  # your Retrieval class should already use the CosmosDB connection

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

# ===================== FUNCTIONS ===================== #
def get_response(user_query: str, top_k: int = 5) -> str:
    """
    Fetch top PDF chunks and generate AI response using Azure OpenAI chat deployment.
    """
    try:
        # 1️⃣ Retrieve relevant chunks
        top_chunks = retriever.search_similar_chunks(user_query, top_k=top_k)
        if not top_chunks:
            return "⚠️ Sorry, I could not find relevant information in the PDF."

        context = "\n\n".join([c["text"] for c in top_chunks])

        # 2️⃣ Build prompt
        prompt = f"""
You are a helpful mobile store assistant.
Use the context below to answer user's query concisely and accurately.

Context:
{context}

Question: {user_query}
Answer:
"""

        # 3️⃣ Call Azure OpenAI chat completion
        response = client.chat.completions.create(
            model=AUTOGEN_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error generating response: {e}"
 