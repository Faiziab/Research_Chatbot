# Research Chatbot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TinyLlama](https://img.shields.io/badge/LLM-TinyLlama-orange.svg)](https://github.com/jzhang38/TinyLlama)
[![FAISS](https://img.shields.io/badge/Search-FAISS-green.svg)](https://github.com/facebookresearch/faiss)

Research Chatbot is an intelligent document question-answering system that enables natural language queries on PDF documents. It combines the power of TinyLlama-1.1B language model with FAISS similarity search to provide accurate, context-aware responses to questions about document content.

## Features

- **Semantic Search**: Utilizes FAISS (Facebook AI Similarity Search) for efficient document retrieval
- **Context-Aware Responses**: Generates answers considering the full context of your documents
- **PDF Processing**: Robust text extraction and intelligent document chunking
- **Interactive Streamlit Interface**: User-friendly web application for document queries
- **GPU Acceleration**: Supports efficient model inference

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Research_Chatbot.git
cd Research_Chatbot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the Streamlit app
streamlit run app.py
```

## System Architecture

```plaintext
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   PDF Doc   │────>│  Text Chunks │────>│ FAISS Index │
└─────────────┘     └──────────────┘     └─────────────┘
                                               │
┌─────────────┐     ┌──────────────┐          │
│    Query    │────>│ LLM Response │<─────────┘
└─────────────┘     └──────────────┘
```

## Components

- `app.py`: Streamlit web interface for Research Chatbot
- `PDFChatbot.py`: Core implementation of PDF processing and question-answering logic
- `requirements.txt`: Project dependencies

## Technical Details

### Language Model
- **Model**: TinyLlama-1.1B-Chat-v1.0
- **Features**: 
  - Half-precision (FP16) inference
  - Automatic device mapping
  - Low CPU memory usage

### Vector Search
- **Engine**: FAISS
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Features**:
  - Dense vector similarity search
  - Efficient nearest neighbor computation

## Usage

1. Launch the Streamlit application
2. Upload a PDF file
3. Ask questions about the document's content

## Usage Example

```
# In the Streamlit interface
1. Click on "Upload a PDF" button
2. Select your PDF file
3. Type your question in the text input field
4. Receive context-aware answers
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Streamlit
- FAISS
- Sentence Transformers
- PyPDF2

For detailed requirements, see `requirements.txt`

## Performance Optimization

- Utilizes GPU acceleration when available
- Implements efficient text chunking for large documents
- Optimizes memory usage through model selection

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- [TinyLlama](https://github.com/jzhang38/TinyLlama) for the language model
- [Facebook Research](https://github.com/facebookresearch/faiss) for FAISS
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Streamlit](https://streamlit.io/) for the interactive web interface

## Contact

Faiziab Khan - [@FaiziabKhan](https://www.linkedin.com/in/faiziab-k-1a3a26121/) - faiziabkhan1@gmail.com

Project Link: [https://github.com/Faiziab/Research_Chatbot.git](Research_Chatbot)
