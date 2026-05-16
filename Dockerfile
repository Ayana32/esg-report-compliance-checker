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

# Copy source code and GRI checklists
COPY *.py .
COPY data/checklists/ data/checklists/

# API key injected at runtime via .env or docker run -e
ENV OPENAI_API_KEY=""

# Streamlit default port
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
