import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import requests
from pymongo import MongoClient
from config.settings import (
    COSMOS_CONNECTION_STRING,
    COSMOS_DATABASE_NAME,
    COSMOS_COLLECTION_NAME
)

# Setup
st.set_page_config(page_title="📱 Mobile Store Chatbot", layout="wide")
mongo_client = MongoClient(COSMOS_CONNECTION_STRING)
db = mongo_client[COSMOS_DATABASE_NAME]
chat_collection = db[COSMOS_COLLECTION_NAME]

st.title("Mobile Store Chatbot")
st.write("Ask a question about mobiles, accessories, offers, or services!")

# Sidebar
st.sidebar.header("Chat History")
if st.sidebar.button("Clear Chat History"):
    chat_collection.delete_many({})
    st.sidebar.success("Chat history cleared!")

history = list(chat_collection.find().sort("_id", -1).limit(20))
if history:
    for msg in reversed(history):
        st.sidebar.write(f"- {msg.get('content', '')[:50]}{'...' if len(msg.get('content', '')) > 50 else ''}")
else:
    st.sidebar.write("No chat history yet.")

# User input
if "input" not in st.session_state:
    st.session_state["input"] = ""

with st.form("query_form", clear_on_submit=True):
    user_query = st.text_input("Your question:", key="input")
    submit = st.form_submit_button("Ask")

if submit and user_query.strip():
    with st.spinner("Thinking..."):
        try:
            # Send user query to Flask API
            response = requests.post(
                "http://localhost:5000/chat",
                json={"query": user_query}
            )
            if response.status_code == 200:
                answer = response.json().get("response", "⚠️ No response.")
            else:
                answer = "⚠️ Server error. Please try again."
        except Exception as e:
            answer = f"❌ Error connecting to backend: {e}"

    st.markdown(f"**You:** {user_query}")
    st.markdown(f"**Bot:** {answer}")

    # Store user query in MongoDB
    chat_collection.insert_one({"role": "user", "content": user_query})
