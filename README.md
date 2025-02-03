

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek_R1-orange.svg)](https://github.com/deepseek-ai)
[![FAISS](https://img.shields.io/badge/Search-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research_Chatbot is an advanced document question-answering system that enables natural language queries on PDF documents. It combines the power of DeepSeek-R1-1.5B language model with FAISS similarity search to provide accurate, context-aware responses to questions about document content.

## Features

- **Semantic Search**: Utilizes FAISS (Facebook AI Similarity Search) for efficient document retrieval
- **Context-Aware Responses**: Generates answers considering the full context of your documents
- **PDF Processing**: Robust text extraction and intelligent document chunking
- **Optimized Performance**: GPU acceleration support and efficient memory management
- **Interactive Interface**: User-friendly command-line interface for document queries

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Research_chatbot.git
cd Research_chatbot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from main import interactive_session

# Start querying your document
interactive_session("path_to_your_document.pdf")
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

- `main.py`: Orchestrates the entire question-answering pipeline
- `faiss_utils.py`: Handles document indexing and similarity search
- `model_utils.py`: Manages the DeepSeek LLM for response generation
- `pdf_utils.py`: Processes PDF documents and extracts text

## Technical Details

### Language Model
- **Model**: DeepSeek-R1-1.5B
- **Features**: 
  - Half-precision (FP16) inference
  - Automatic device mapping
  - Optimized beam search
  - Memory-efficient processing

### Vector Search
- **Engine**: FAISS
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Features**:
  - Dense vector similarity search
  - Efficient nearest neighbor computation
  - Scalable document indexing

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- FAISS
- PyMuPDF
- Sentence Transformers

For detailed requirements, see [requirements.txt](requirements.txt)

## Usage Example

```bash
$ python main.py
Enter your question (or type 'quit' to exit): What are the main findings of the paper?

Processing your question...
1. Extracting text from PDF...
2. Chunking text...
3. Creating search index...
4. Finding relevant passages...
5. Generating detailed answer...

Answer: [Generated response based on document content]
```

## Performance Optimization

- Utilizes GPU acceleration when available
- Implements efficient text chunking for large documents
- Optimizes memory usage through model quantization
- Employs caching for frequent queries

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- [DeepSeek AI](https://github.com/deepseek-ai) for the language model
- [Facebook Research](https://github.com/facebookresearch/faiss) for FAISS
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)

## Contact

Your Name - [@FaiziabKhan](https://www.linkedin.com/in/faiziab-k-1a3a26121/) - faiziabkhan1@gmail.com

Project Link: [https://github.com/Faiziab/Research_Chatbot.git](Research_Chatbot)