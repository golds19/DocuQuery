# Loading the libraries
from dotenv import load_dotenv
import pymupdf
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import tempfile
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from src.ingestion import PdfExtractors
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings

# Load environment variables
load_dotenv()

def validate_vector_store_config():
    """Validate vector store configuration and provide helpful error messages"""
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError(
            "PINECONE_API_KEY not found. Please create a Pinecone account and "
            "set your API key in the .env file. Visit https://www.pinecone.io to get started."
        )
    
    # Validate OpenAI configuration
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set your OpenAI API key in the .env file "
            "if you plan to use OpenAI models."
        )

def get_index_name(model_choice):
    """Get user-provided index name or raise helpful error"""
    index_env_var = f"{model_choice.upper().replace(' ', '_')}_INDEX_NAME"
    index_name = os.getenv(index_env_var)
    if not index_name:
        raise ValueError(
            f"Please set {index_env_var} in your .env file. "
            "This should be your own Pinecone index name."
        )
    return index_name

# Validate configuration on startup
validate_vector_store_config()

# Get index names through the new function
def get_vector_store(model_name: str, namespace: str = None):
    """
    Get vector store with proper configuration and namespace
    
    Args:
        model_name: The model choice ("OpenAI" or "Llama 3")
        namespace: Optional namespace for data isolation
    
    Returns:
        PineconeVectorStore: Configured vector store instance
    """
    # Get the appropriate index name
    index_name = get_index_name(model_name)
    
    # Set up embeddings based on model choice
    embeddings = (OpenAIEmbeddings(model="text-embedding-3-small") 
                 if model_name == "OpenAI" 
                 else OllamaEmbeddings(model="mxbai-embed-large"))
    
    # Create vector store with optional namespace
    return PineconeVectorStore(
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace
    )

# Set Tesseract and Poppler paths (adjust as needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# POPPLER_PATH = r'C:\Poppler\bin'

# Pinecone index name
OPENAI_INDEX_NAME = os.getenv("OPENAI_INDEX_NAME")
OLLAMA_INDEX_NAME = os.getenv("OLLAMA_INDEX_NAME")

def upload_files(files, upload_dir="uploaded_files"):
    """
    Process files - handles both file paths and Streamlit uploads
    """
    pdf_paths = []
    image_paths = []
    
    for file_item in files:
        try:
            if isinstance(file_item, str):
                # It's a file path
                file_path = file_item
                file_ext = os.path.splitext(file_item)[1].lower()
            else:
                # It's a Streamlit UploadedFile
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, file_item.name)
                file_ext = os.path.splitext(file_item.name)[1].lower()
                
                # Save the uploaded file
                with open(file_path, 'wb') as f:
                    f.write(file_item.getbuffer())
            
            # Categorize
            if file_ext == '.pdf':
                pdf_paths.append(file_path)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                image_paths.append(file_path)
                
        except Exception as e:
            print(f"Error processing file: {e}")
            continue
    
    return pdf_paths, image_paths

def categorize_file_paths(file_paths):
    """
    Categorize existing file paths into PDFs and images.
    
    Args:
        file_paths: List of file path strings
    
    Returns:
        Tuple[List[str], List[str]]: PDF paths and image paths
    """
    pdf_paths = []
    image_paths = []
    
    for file_path in file_paths:
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            # Get file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Categorize based on extension
            if file_ext == '.pdf':
                pdf_paths.append(file_path)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                image_paths.append(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return pdf_paths, image_paths

def load_and_chunk_pdfs_and_images(pdf_paths, image_paths, chunk_size=500, chunk_overlap=50):
    """
    Load, extract text from PDFs and images, and chunk into Document objects.
    
    Args:
        pdf_paths: List of paths to PDF files.
        image_paths: List of paths to image files.
        chunk_size: Size of each text chunk (characters).
        chunk_overlap: Overlap between chunks (characters).
    
    Returns:
        List[Document]: List of chunked Document objects.
    """
    documents = []

    # Process PDFs
    for pdf_path in pdf_paths:
        pdf_extractor = PdfExtractors(pdf_path)
        try:
            pdf_text = pdf_extractor.extract_text_from_pdf()
            if pdf_text and not pdf_text.startswith("Error"):
                documents.append(Document(page_content=pdf_text, metadata={"source": pdf_path, "type": "pdf"}))
            else:
                pdf_text_from_images = pdf_extractor.extract_text_from_images()
                if not pdf_text_from_images.startswith("Error"):
                    documents.append(Document(page_content=pdf_text_from_images, metadata={"source": pdf_path, "type": "pdf_images"}))
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")

    # Process images
    for image_path in image_paths:
        try:
            pdf_extractor = PdfExtractors("")  # Dummy path for extract_from_jpg
            image_text = pdf_extractor.extract_from_jpg(image_path)
            if not image_text.startswith("Error"):
                documents.append(Document(page_content=image_text, metadata={"source": image_path, "type": "jpg"}))
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    # Chunk documents
    chunked_documents = []
    for doc in documents:
        text = doc.page_content
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            chunked_documents.append(Document(page_content=chunk, metadata=doc.metadata))

    return chunked_documents

def documents_storing_in_vectorDB(documents, model_name, namespace=None):     
    """     
    Store documents in Pinecone vector store, with automatic batch processing for long documents.          
    Args:         
        documents: List of Document objects to store.         
        model_name: The model choice ("OpenAI" or "Llama 3")
        namespace: Optional namespace for data isolation
    Returns:         
        PineconeVectorStore: Initialized vector store for querying.     
    """     
    # Get configured vector store
    vectorstore = get_vector_store(model_name, namespace)
    
    # Analyze document lengths to determine processing method
    total_chars = sum(len(doc.page_content) for doc in documents)
    avg_doc_length = total_chars / len(documents) if documents else 0
    
    # Thresholds for automatic batch processing
    total_threshold = 100000    # Total characters across all documents
    avg_threshold = 2000       # Average characters per document
    
    use_batch_processing = (
        total_chars > total_threshold or 
        avg_doc_length > avg_threshold
    )
    
    if use_batch_processing:
        print(f"Documents are long (total: {total_chars:,} chars, avg: {avg_doc_length:.0f} chars/doc)")
        print("Using batch processing for optimal performance...")
        
        batch_size = 50  # Smaller batches for long documents
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectorstore.add_documents(batch)
            print(f"Uploaded batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size} with {len(batch)} documents")
        
        print("****Loading to vectorstore done ****")
        return vectorstore
    
    else:
        print(f"Documents are manageable (total: {total_chars:,} chars, avg: {avg_doc_length:.0f} chars/doc)")
        print("Using standard processing...")
        
        # Standard processing for shorter documents
        print(f"Going to add {len(documents)} documents to Pinecone")         
        vectorstore.add_documents(documents)
        print(f"Documents stored successfully in index: {get_index_name(model_name)}")         
        return vectorstore

def process_uploaded_files(uploaded_files, model_choice, namespace=None):
    """
    Process uploaded files, extract text, chunk, and store in Pinecone.
    
    Args:
        uploaded_files: List of file-like objects from UI upload.
        model_choice: Selected model type (OpenAI or Llama 3).
        namespace: Optional namespace for data isolation
    Returns:
        PineconeVectorStore: Vector store for querying, or None if failed.
    """
    try:
        # Save uploaded files to temporary paths
        pdf_paths, image_paths = upload_files(uploaded_files)
        print(pdf_paths)
        print(image_paths)
        
        # Process and chunk documents
        documents = load_and_chunk_pdfs_and_images(pdf_paths, image_paths)
        print(documents)
        
        # Store in Pinecone with namespace if provided
        if documents:
            vectorstore = documents_storing_in_vectorDB(
                documents, 
                model_name=model_choice,
                namespace=namespace
            )
            return vectorstore
        else:
            print("No valid documents extracted")
            return None
    except Exception as e:
        print(f"Error occurred: {e}")
        raise

def run_llm_query(query: str, model_choice: str, namespace: str = None):
    """
    Run a query against the Pinecone store using RAG pipeline.
    
    Args:
        query: String query from user.
        model_choice: Selected model type (OpenAI or Llama 3).
        namespace: Optional namespace for data isolation
    
    Returns:
        Dict: Result from RAG pipeline, including answer.
    """
    
    try:
        # Get configured vector store with namespace
        docsearch = get_vector_store(model_choice, namespace)
        
        # Set up the retrieval chain
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        
        # Create the stuff documents chain
        stuff_documents_chain = create_stuff_documents_chain(
            ChatOpenAI(verbose=True, temperature=0, model="gpt-4o-mini-2024-07-18") 
            if model_choice == "OpenAI" 
            else OllamaLLM(model="llama3:8b"), 
            retrieval_qa_chat_prompt
        )
        
        # Create the QA chain
        qa = create_retrieval_chain(
            retriever=docsearch.as_retriever(),
            combine_docs_chain=stuff_documents_chain
        )
        
        # Run the query
        result = qa.invoke({"input": query})
        return result
    except Exception as e:
        print(f"Error occurred: {e}")
        raise

# Example usage (for testing without UI)
if __name__ == "__main__":
    # Simulate uploaded files with local paths
    pdf_paths = [
        "Data/adobe.pdf",
    ]
    image_paths = [
        "Data/imagetest1.jpg",
        "Data/imagetest2.jpg"
    ]
    
    # Process "uploaded" files
    vectorstore = process_uploaded_files(pdf_paths + image_paths, model_choice="OpenAI")
    
    # Test query
    result = run_llm_query("What are the AI and digital trends in 2025 from the uploaded documents?", "OpenAI")
    print(result["answer"])