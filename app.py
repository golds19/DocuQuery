from src.main import process_uploaded_files, run_llm_query, validate_vector_store_config
import streamlit as st
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Validate configuration on startup
try:
    validate_vector_store_config()
except ValueError as e:
    st.error(f"Configuration Error: {str(e)}")
    st.stop()

st.title("OCR RAG Application")

# Add a section for namespace (optional)
st.sidebar.markdown("### Advanced Settings")
use_namespace = st.sidebar.checkbox("Use Custom Namespace")
namespace = None
if use_namespace:
    namespace = st.sidebar.text_input(
        "Enter Namespace",
        help="Optional: Enter a namespace to isolate your data"
    )

st.markdown("### Upload Documents")
uploaded_files = st.file_uploader("Upload PDFs and Images", accept_multiple_files=True, type=['pdf', 'jpg', 'jpeg', 'png'])

# Add model selection dropdown at the same level as upload documents
model_choice = st.selectbox("Select Model:", ("OpenAI", "Llama 3"))

# Placeholder for success message
success_message = st.empty()

if uploaded_files and st.button("Process Documents"):
    with st.spinner("Processing documents..."):
        progress_bar = st.progress(0)
        try:
            vectorstore = process_uploaded_files(
                uploaded_files, 
                model_choice=model_choice,
                namespace=namespace
            )
            st.session_state.vectorstore = vectorstore  # Store for querying
            progress_bar.progress(100)  # Complete progress
            success_message.success("Documents processed and saved in vector store!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
        # Clear the success message after a short delay
        time.sleep(2)
        success_message.empty()

# Ask Your Question
st.markdown("### Ask Your Question")

# Use session state to manage the query input
if 'query' not in st.session_state:
    st.session_state.query = ""

query = st.text_input("Type your question here:", value=st.session_state.query)

if query and st.button("Submit Query"):
    # Clear the query input first
    st.session_state.query = ""
    with st.spinner("Submitting query..."):
        progress_bar = st.progress(0)
        try:
            result = run_llm_query(
                query, 
                model_choice,
                namespace=namespace
            )
            progress_bar.progress(100)  # Complete progress
            st.write("Answer:", result["answer"])
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")