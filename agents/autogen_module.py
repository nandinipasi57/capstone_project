import os
from dotenv import load_dotenv
from retrieve_documents import Retrieval  # Updated import path
from openai import AzureOpenAI

# Load .env file
load_dotenv()

# ===================== CONFIG ===================== #
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AUTOGEN_DEPLOYMENT = os.getenv("AUTOGEN_DEPLOYMENT")

# ===================== ASSISTANT AGENT CLASS ===================== #
class AssistantAgent:
    def __init__(self, name, model_client, system_message):
        self.name = name
        self.model_client = model_client
        self.system_message = system_message

    def get_response(self, user_query, context):
        prompt = f"""
{self.system_message}
Context:
{context}

Question: {user_query}
Answer:
"""
        try:
            # Assuming model_client is an OpenAI client (could be other models as well)
            response = self.model_client.chat.completions.create(
                model="gpt-3.5-turbo",  # or any model you're using
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error from AssistantAgent: {e}")
            return None

# ===================== INITIALIZE MODULES ===================== #

# Initialize Retrieval module (fetches from Cosmos DB)
retriever = Retrieval()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

# Initialize Assistant Agent
assistant_agent = AssistantAgent(
    name="mobile_store_assistant",
    model_client=client,  # Use the Azure OpenAI client for the model
    system_message="You are a helpful mobile store assistant. Answer concisely using any retrieved context."
)

# ===================== FUNCTIONS ===================== #
def get_response(user_query: str, top_k: int = 5) -> str:
    """
    Fetch top PDF chunks, use AssistantAgent for response, and fallback to Azure OpenAI if needed.
    """

    try:
        # 1️⃣ Retrieve relevant chunks from Cosmos DB
        top_chunks = retriever.search_similar_chunks(user_query, top_k=top_k)
        print(f"[DEBUG] Query: {user_query}")
        print(f"[DEBUG] Retrieved Chunks: {top_chunks}")

        if top_chunks:
            context = "\n\n".join([c["text"] for c in top_chunks])
        else:
            context = ""  # No context found

        # 2️⃣ Build a prompt using retrieved context and the user's query
        prompt = f"""
You are a helpful mobile store assistant.
Use the context below to answer the user's query concisely and accurately.
Do not answer if the question is unrelated to the context.
Answer in a professional manner and check the database carefully before answering.

Context:
{context}

Question: {user_query}
Answer:
"""

        # 3️⃣ Call AssistantAgent to process the prompt
        agent_response = assistant_agent.get_response(user_query, context)
        if agent_response:
            return agent_response  # Return response from AssistantAgent

        # 4️⃣ Fallback: If AssistantAgent doesn't respond, call Azure OpenAI chat completion to get the response
        response = client.chat.completions.create(
            model=AUTOGEN_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "❌ An error occurred while processing your request. Please try again later."
 