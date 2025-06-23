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
from ingestion import PdfExtractors

# Load environment variables (e.g., PINECONE_API_KEY, OPENAI_API_KEY)
load_dotenv()

# Set Tesseract and Poppler paths (adjust as needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# POPPLER_PATH = r'C:\Poppler\bin'

# Set embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Pinecone index name
INDEX_NAME = "ocr-rag"

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

def documents_storing_in_vectorDB(documents):
    """
    Store documents in Pinecone vector store, with batch processing fallback.
    
    Args:
        documents: List of Document objects to store.
    
    Returns:
        PineconeVectorStore: Initialized vector store for querying.
    """
    try:
        # Try storing all documents at once
        print(f"Going to add {len(documents)} to Pinecone")
        vectorstore = PineconeVectorStore.from_documents(
            documents,
            embeddings,
            index_name=INDEX_NAME
        )
        print(f"Documents stored successfully")
        return vectorstore
    except Exception as e:
        print(f"Documents storing failed with from_documents: {e}. Switching to batch processing...")
        try:
            # Fallback to batch processing
            vectorstore = PineconeVectorStore(
                embedding=embeddings,
                index_name=INDEX_NAME
            )
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                vectorstore.add_documents(batch)
                print(f"Uploaded batch {i // batch_size + 1} with {len(batch)} documents")
            print("****Loading to vectorstore done ****")
            return vectorstore
        except Exception as batch_error:
            print(f"Documents storing failed with batch processing: {batch_error}")
            raise

def process_uploaded_files(uploaded_files):
    """
    Process uploaded files, extract text, chunk, and store in Pinecone.
    
    Args:
        uploaded_files: List of file-like objects from UI upload.
    Returns:
    PineconeVectorStore: Vector store for querying, or None if failed.
    """
    try:
        # Save uploaded files to temporary paths
        pdf_paths, image_paths = categorize_file_paths(uploaded_files)
        print(pdf_paths)
        print(image_paths)
        
        # Process and chunk documents
        documents = load_and_chunk_pdfs_and_images(pdf_paths, image_paths)
        print(documents)
        
        # Store in Pinecone
        if documents:
            vectorstore = documents_storing_in_vectorDB(documents)
            return vectorstore
        else:
            print("No valid documents extracted")
            return None
    except Exception as e:
        print(f"Error occured: {e}")

def run_llm_query(query: str):
    """
    Run a query against the Pinecone store using RAG pipeline.
    
    Args:
        query: String query from user.
        vectorstore: PineconeVectorStore instance for retrieval.
    
    Returns:
        Dict: Result from RAG pipeline, including answer.
    """
    # if not vectorstore:
    #     return {"answer": "No documents have been processed yet. Please upload documents first."}
    
    chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-4o-mini-2024-07-18")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    
    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(),
        combine_docs_chain=stuff_documents_chain
    )
    
    result = qa.invoke({"input": query})
    return result

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
    # vectorstore = process_uploaded_files(pdf_paths + image_paths)
    
    # Test query
    result = run_llm_query("What are the AI and digital trends in 2025 from the uploaded documents?")
    print(result["answer"])