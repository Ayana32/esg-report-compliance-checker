FROM python:3.11-slim

# Working directory
WORKDIR /app

# System packages (required for ChromaDB and PyMuPDF)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Copy application source, data, and vector store
COPY app/ app/
COPY *.py ./
COPY data/ data/
COPY chroma_db/ chroma_db/

# FastAPI default port
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]