from src.main import process_uploaded_files, run_llm_query
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.title("OCR RAG Application")
uploaded_files = st.file_uploader("Upload PDFs and Images", accept_multiple_files=True, type=['pdf', 'jpg', 'jpeg', 'png'])

if uploaded_files and st.button("Process Documents"):
    with st.spinner("Processing documents..."):
        vectorstore = process_uploaded_files(uploaded_files)
        st.session_state.vectorstore = vectorstore  # Store for querying
    st.success("Documents processed and stored in Pinecone!")

# Query interface
if 'vectorstore' in st.session_state:
    query = st.text_input("Ask a question about the documents")
    if query and st.button("Submit Query"):
        result = run_llm_query(query)
        st.write("Answer:", result["answer"])