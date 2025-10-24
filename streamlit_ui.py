# streamlit_ui.py
import streamlit as st
from autogen_module import get_response
from pymongo import MongoClient
from config.settings import (
    COSMOS_CONNECTION_STRING,
    COSMOS_DATABASE_NAME,
    COSMOS_COLLECTION_NAME
)

# -------------------- SETUP -------------------- #
st.set_page_config(page_title="Mobile Store Chatbot", layout="wide")

# CosmosDB client
mongo_client = MongoClient(COSMOS_CONNECTION_STRING)
db = mongo_client[COSMOS_DATABASE_NAME]
chat_collection = db[COSMOS_COLLECTION_NAME]

st.title("Mobile Store Chatbot")
st.write("Ask a question about our mobile products:")

# -------------------- SIDEBAR: Chat History -------------------- #
st.sidebar.header("Chat History")
# Fetch last 20 user messages to display
history = list(chat_collection.find({"role": "user"}).sort("_id", -1).limit(20))
if history:
    for msg in reversed(history):
        # Show truncated title
        st.sidebar.write(f"- {msg['content'][:50]}{'...' if len(msg['content']) > 50 else ''}")
else:
    st.sidebar.write("No chat history yet.")

# -------------------- USER INPUT -------------------- #
user_query = st.text_input("Your question:")

if st.button("Ask") and user_query.strip():
    with st.spinner("Thinking..."):
        answer = get_response(user_query)

    # Display conversation in main page
    st.markdown(f"**You:** {user_query}")
    st.markdown(f"**Bot:** {answer}")

    # Store chat in CosmosDB
    chat_collection.insert_one({"role": "user", "content": user_query})
    chat_collection.insert_one({"role": "bot", "content": answer})
 