# faiss_utils.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, chunk_size=512):
    """
    Split the document text into chunks of specified max size.
    """
    chunks = text.split(". ")  # Split by period and space for sentence chunks
    return [chunk[:chunk_size] for chunk in chunks]

def create_faiss_index(text_chunks):
    """
    Create a FAISS index from the provided text chunks.
    """
    embeddings = embedder.encode(text_chunks)
    embeddings = np.array(embeddings).astype(np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for similarity
    index.add(embeddings)
    return index

def retrieve_top_k_passages(query, index, text_chunks, k=5):
    """
    Retrieve the top k most relevant passages from the FAISS index based on the user's query.
    """
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype(np.float32)
    
    # Retrieve top k passages using FAISS
    D, I = index.search(query_embedding, k)  # D = distances, I = indices of nearest neighbors
    return [text_chunks[i] for i in I[0]]
