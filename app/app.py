# app.py
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import HuggingFacePipeline
from transformers import pipeline
import os

# --- Set up session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Load Hugging Face embeddings ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Load your vectorstore (replace with your persisted DB if any) ---
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# --- Load Hugging Face model via pipeline ---
hf_pipe = pipeline(
    task="text2text-generation",  # or "text-generation" for plain LLM
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    device=-1  # -1 CPU, 0 GPU if available
)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# --- Setup conversational chain ---
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# --- Streamlit interface ---
st.title("Story AI Assistant")
user_q = st.text_input("Ask me about your story:")

if user_q:
    result = qa_chain({"question": user_q, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((user_q, result["answer"]))
    st.write(result["answer"])
    # Optional: show sources
    for doc in result["source_documents"]:
        st.write(f"Source: {doc.metadata.get('source', 'unknown')}")
