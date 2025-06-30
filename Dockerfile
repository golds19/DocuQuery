# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Tesseract and Poppler
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p uploaded_files Data

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables for Tesseract and Poppler
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV POPPLER_PATH=/usr/bin

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"] 