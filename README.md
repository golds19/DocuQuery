# OCR RAG Document Q&A

A Python-based application that processes PDFs and images using OCR, stores text embeddings in Pinecone, and answers user queries via a Retrieval-Augmented Generation (RAG) pipeline. Users can upload documents through a Streamlit UI, extract text, and query insights using natural language.


## Features
- **OCR Processing**: Extracts text from PDFs and images using Tesseract and PyMuPDF.
- **Text Chunking**: Splits documents into manageable chunks for efficient retrieval.
- **Vector Storage**: Stores embeddings in Pinecone for fast similarity search.
- **RAG Pipeline**: Combines LangChain, OpenAI embeddings, and GPT-4o-mini for accurate query responses.
- **Streamlit UI**: User-friendly interface for document uploads and querying.
- **Batch Processing**: Handles large document sets with fallback batch uploads to Pinecone.

## Tech Stack
- **Languages**: Python
- **OCR**: Tesseract, PyMuPDF, pdf2image
- **NLP**: LangChain, OpenAI (text-embedding-3-small, GPT-4o-mini)
- **Vector DB**: Pinecone
- **Frontend**: Streamlit
- **Dependencies**: Pillow, python-dotenv


## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/golds/ocr-rag.git
   cd ocr-rag
Set Up Virtual Environment:
bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install Dependencies:
bash
pip install -r requirements.txt
Install System Dependencies:
Tesseract: Download and install Tesseract. Add to PATH (e.g., C:\Program Files\Tesseract-OCR).
Poppler: Download Poppler and add bin to PATH (e.g., C:\Poppler\bin).
Set Environment Variables:
Create a .env file in the root directory:
env
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key
Run the Application:
bash
streamlit run app.py
Usage
Open the Streamlit app in your browser (default: http://localhost:8501).
Upload PDFs or images (.pdf, .jpg, .png).
Click "Process Documents" to extract text and store in Pinecone.
Enter a query (e.g., "What is the global revenue of artificial intelligence?") to get answers based on uploaded documents.

### Project Folder Architecture
├── src/                # Source code
│   ├── ingestion.py   # PdfExtractors class for OCR
│   ├── main.py        # Core logic for document processing, vector storage, and RAG querying
├── data/              # Sample PDFs and images (not tracked)
├── docs/              # Documentation and screenshots
├── notebooks/         # Jupyter notebooks for experiments and analysis
├── app.py             # Streamlit UI for document uploads and querying
├── requirements.txt   # Python dependencies
├── packages.txt       # System dependencies for deployment
├── .env               # Environment variables (not tracked)
├── .gitignore        # Git ignore patterns
└── README.md         # Project overview