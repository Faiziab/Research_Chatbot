# Research Chatbot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TinyLlama](https://img.shields.io/badge/LLM-TinyLlama-orange.svg)](https://github.com/jzhang38/TinyLlama)
[![FAISS](https://img.shields.io/badge/Search-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)

Research Chatbot is an intelligent document question-answering system that enables natural language interaction with PDF documents. Built with TinyLlama-1.1B and FAISS similarity search, it provides accurate, context-aware responses to questions about document content. The application features a user-friendly Streamlit interface and is optimized for cloud deployment.

## 🌟 Features

- **PDF Processing**: Advanced text extraction and intelligent document chunking
- **Semantic Search**: FAISS-powered similarity search for precise context retrieval
- **Context-Aware Responses**: AI-generated answers based on document context
- **Interactive Interface**: Clean, responsive Streamlit web application
- **Cloud Ready**: Optimized for deployment on cloud platforms
- **Memory Efficient**: Optimized for resource-constrained environments

## 🚀 Live Demo

Try the live application: [Research Chatbot Demo](https://your-app-url.streamlit.app)

## 💻 Installation

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

## 🏃‍♂️ Quick Start

```bash
# Run locally
streamlit run app.py
```

## 🏗️ System Architecture

```plaintext
📄 PDF Document → 📝 Text Chunks → 🔍 FAISS Index
                                        ↓
❓ User Query → 🤖 LLM Processing → 💬 Response
```

## 🔧 Components

- `app.py`: Streamlit web interface
- `researchchatbot.py`: Core chatbot implementation
- `requirements.txt`: Project dependencies

## 🛠️ Technical Stack

### Language Model
- **Model**: TinyLlama-1.1B-Chat-v1.0
- **Features**: 
  - FP16 inference
  - Automatic device mapping
  - Memory-optimized implementation

### Vector Search
- **Engine**: FAISS
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)

## 📖 Usage Guide

1. Open the application (local or deployed version)
2. Upload your PDF document using the sidebar
3. Wait for processing completion
4. Type your questions in the chat interface
5. Receive AI-generated responses based on document content

## ⚙️ Requirements

```plaintext
Python 3.8+
streamlit
torch
transformers
sentence-transformers
PyPDF2
faiss-cpu
numpy
rich
```

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Fork this repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Deploy directly from your GitHub repository

### Alternative Platforms
- **Hugging Face Spaces**: Deploy via [huggingface.co/spaces](https://huggingface.co/spaces)
- **Render**: Deploy using [render.com](https://render.com)

## ⚡ Performance Tips

- Use CPU-optimized versions of dependencies for cloud deployment
- Monitor memory usage in cloud environments
- Consider file size limitations of deployment platforms
- Implement caching for better performance

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

- [TinyLlama](https://github.com/jzhang38/TinyLlama) for the language model
- [Facebook Research](https://github.com/facebookresearch/faiss) for FAISS
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Streamlit](https://streamlit.io/) for the interactive web interface

## Contact

Faiziab Khan - [@FaiziabKhan](https://www.linkedin.com/in/faiziab-k-1a3a26121/) - faiziabkhan1@gmail.com

Project Link: [https://github.com/Faiziab/Research_Chatbot.git](Research_Chatbot)