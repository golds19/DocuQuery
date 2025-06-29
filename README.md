# OCR RAG Document Q&A

A Python-based application that processes PDFs and images using OCR, stores text embeddings in Pinecone, and answers user queries via a Retrieval-Augmented Generation (RAG) pipeline. Users can upload documents through a Streamlit UI, extract text, and query insights using natural language.

## Features

- **OCR Processing**: Extracts text from PDFs and images using Tesseract and PyMuPDF
- **Text Chunking**: Splits documents into manageable chunks for efficient retrieval
- **Vector Storage**: Stores embeddings in Pinecone for fast similarity search
- **RAG Pipeline**: Combines LangChain, OpenAI embeddings, and GPT-4o-mini for accurate query responses
- **Streamlit UI**: User-friendly interface for document uploads and querying
- **Batch Processing**: Handles large document sets with fallback batch uploads to Pinecone

- **Model Integration**: The application now supports both OpenAI and Llama 3 models for document processing and querying.
- **Dynamic Indexing**: The index name used for storing documents in Pinecone is now determined based on the selected model.

## Tech Stack

- **Languages**: Python
- **OCR**: Tesseract, PyMuPDF, pdf2image
- **NLP**: LangChain, OpenAI (text-embedding-3-small, GPT-4o-mini), Llama3:8b (mxbai-embed-large)
- **Vector DB**: Pinecone
- **Frontend**: Streamlit
- **Dependencies**: Pillow, python-dotenv

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/golds19/DocuQuery.git
cd ocr-rag
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install System Dependencies

**Tesseract:**
- Download and install [Tesseract](https://github.com/tesseract-ocr/tesseract)
- Add to PATH (e.g., `C:\Program Files\Tesseract-OCR`)

**Poppler:**
- Download [Poppler](https://poppler.freedesktop.org/) and add `bin` to PATH (e.g., `C:\Poppler\bin`)

### 5. Set Environment Variables

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key
```

### 6. Run the Application

```bash
streamlit run app.py
```

## Usage

1. Open the Streamlit app in your browser (default: http://localhost:8501)
2. Upload PDFs or images (`.pdf`, `.jpg`, `.png`)
3. Click "Process Documents" to extract text and store in Pinecone
4. Enter a query (e.g., "What is the global revenue of artificial intelligence?") to get answers based on uploaded documents

## Project Structure

```
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
├── .gitignore         # Git ignore patterns
└── README.md          # Project overview
```

## Requirements

- Python 3.8+
- Tesseract OCR
- Poppler (for PDF processing)
- Pinecone account
- OpenAI API key

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
