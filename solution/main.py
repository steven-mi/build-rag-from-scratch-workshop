import streamlit as st

from embedding_db import EmbeddingDatabase
from embedding_model import EmbeddingModel
from llm import LargeLanguageModel
from rag import QuestionAnsweringRAG


# Initialize models and RAG
embedding_model = EmbeddingModel()
embedding_db = EmbeddingDatabase(embedding_model)
llm = LargeLanguageModel()
rag = QuestionAnsweringRAG(llm, embedding_db)

# Streamlit app title
st.title("Q&A Food RAG")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input and response handling
if prompt := st.chat_input("What is up?"):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    response = rag.query(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)
