# OCR RAG Document Q&A

A Python-based application that processes PDFs and images using OCR, stores text embeddings in Pinecone, and answers user queries via a Retrieval-Augmented Generation (RAG) pipeline. Users can upload documents through a Streamlit UI, extract text, and query insights using natural language.

## Features

- **OCR Processing**: Extracts text from PDFs and images using Tesseract and PyMuPDF
- **Text Chunking**: Splits documents into manageable chunks for efficient retrieval
- **Vector Storage**: Stores embeddings in Pinecone for fast similarity search
- **RAG Pipeline**: Combines LangChain with choice of:
  - OpenAI (GPT-4o-mini + text-embedding-3-small)
  - Llama 3 (local model via Ollama)
- **Streamlit UI**: User-friendly interface for document uploads and querying
- **Namespace Support**: Organize your documents into separate namespaces
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Prerequisites

Before running the application, you need:

1. **Pinecone Setup**:
   - Create a free account at [Pinecone](https://www.pinecone.io)
   - Create two indexes:
     - One for OpenAI embeddings (e.g., "my-openai-index")
     - One for Llama embeddings (e.g., "my-llama-index")
   - Get your API key from the Pinecone console

2. **OpenAI API Key** (Optional):
   - Only needed if you plan to use OpenAI models
   - Get your API key from [OpenAI](https://platform.openai.com)

## Running the Application

There are two ways to run the application:

### Method 1: Using Docker (Recommended)

1. **Create Environment File**:
   Create a `.env` file with your configuration:
   ```env
   # Required: Pinecone Configuration
   PINECONE_API_KEY=your_pinecone_api_key
   OPENAI_INDEX_NAME=your_openai_index_name
   OLLAMA_INDEX_NAME=your_llama_index_name

   # Optional: OpenAI Configuration (if using OpenAI)
   OPENAI_API_KEY=your_openai_api_key
   ```

2. **Create Docker Compose File**:
   Create a `docker-compose.yml`:
   ```yaml
   version: '3.8'

   services:
     web:
       image: benphil/ocr-rag:latest
       ports:
         - "8501:8501"
       volumes:
         - uploaded_files:/app/uploaded_files
         - data:/app/Data
       env_file:
         - .env
       restart: unless-stopped

     ollama:
       image: ollama/ollama:latest
       ports:
         - "11434:11434"
       volumes:
         - ollama_data:/root/.ollama
       restart: unless-stopped

   volumes:
     uploaded_files:
     data:
     ollama_data:
   ```

3. **Run the Application**:
   ```bash
   docker-compose up
   ```

### Method 2: Local Development

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/golds19/DocuQuery.git
   cd ocr-rag
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install System Dependencies**:
   - **Tesseract**:
     - Windows: Download and install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
     - Linux: `sudo apt-get install tesseract-ocr`
     - Mac: `brew install tesseract`
   
   - **Poppler**:
     - Windows: Download [Poppler](http://blog.alivate.com.au/poppler-windows/)
     - Linux: `sudo apt-get install poppler-utils`
     - Mac: `brew install poppler`

5. **Create Environment File**:
   Create `.env` file as shown in Method 1

6. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the application in your browser (default: http://localhost:8501)
2. Optional: Enable and configure a namespace in the sidebar to organize your documents
3. Upload PDFs or images
4. Select your preferred model (OpenAI or Llama 3)
5. Process the documents
6. Start asking questions about your documents

## Using Namespaces

Namespaces help organize your documents within your Pinecone index. Use cases:
- Separate different projects
- Isolate document types (e.g., financial, legal)
- Maintain development and production data separately

To use namespaces:
1. Enable "Use Custom Namespace" in the sidebar
2. Enter a namespace identifier
3. All operations will be isolated to that namespace

## Requirements

- Python 3.12+
- Tesseract OCR
- Poppler
- Pinecone Account
- OpenAI API key (optional)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
