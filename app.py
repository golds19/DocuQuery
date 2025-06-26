from src.main import process_uploaded_files, run_llm_query
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.title("OCR RAG Application")

st.markdown("### Upload Documents")
uploaded_files = st.file_uploader("Upload PDFs and Images", accept_multiple_files=True, type=['pdf', 'jpg', 'jpeg', 'png'])

# Add model selection dropdown at the same level as upload documents
model_choice = st.selectbox("Select Model:", ("OpenAI", "Llama 3"))

st.markdown("### Ask Your Question")
query = st.text_input("Type your question here:")

if uploaded_files and st.button("Process Documents"):
    with st.spinner("Processing documents..."):
        progress_bar = st.progress(0)
        vectorstore = process_uploaded_files(uploaded_files)
        st.session_state.vectorstore = vectorstore  # Store for querying
        progress_bar.progress(100)  # Complete progress
    st.success("Documents processed and saved in vector store!")

# Query interface
if query and st.button("Submit Query"):
    with st.spinner("Submitting query..."):
        progress_bar = st.progress(0)
        result = run_llm_query(query, model_choice)  # Pass selected model
        progress_bar.progress(100)  # Complete progress
        st.write("Answer:", result["answer"])