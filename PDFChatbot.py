import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
import numpy as np
from typing import List, Tuple
import re

class PDFChatbot:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        # Initialize embeddings model
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

        # Initialize storage
        self.texts = []
        self.faiss_index = None

    def preprocess_pdf(self, pdf_path: str, chunk_size: int = 500, overlap: int = 50):
        """Extract and chunk text from PDF"""
        # Read PDF
        reader = PdfReader(pdf_path)
        raw_text = ""

        # Extract text from each page
        for page in reader.pages:
            raw_text += page.extract_text() + " "

        # Clean text
        raw_text = re.sub(r'\s+', ' ', raw_text).strip()

        # Create overlapping chunks
        chunks = []
        start = 0
        while start < len(raw_text):
            end = start + chunk_size
            if end > len(raw_text):
                end = len(raw_text)
            chunk = raw_text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        self.texts = chunks
        return chunks

    def create_embeddings(self):
        """Create FAISS index from text chunks"""
        # Generate embeddings
        embeddings = self.embed_model.encode(self.texts)

        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)

        # Add embeddings to index
        self.faiss_index.add(np.array(embeddings).astype('float32'))

    def get_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        """Retrieve most relevant text chunks for a query"""
        # Generate query embedding
        query_embedding = self.embed_model.encode([query])

        # Search in FAISS index
        distances, indices = self.faiss_index.search(
            np.array(query_embedding).astype('float32'), k
        )

        # Return relevant chunks
        return [self.texts[i] for i in indices[0]]

    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response using LLM"""
        # Create prompt
        prompt = f"""Context: {' '.join(context)}

Question: {query}

Please provide a concise and relevant answer based on the context above.

Answer:"""

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.2,
            top_p=0.6,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the answer part
        answer = response.split("Answer:")[-1].strip()
        return answer

    def chat(self, query: str) -> str:
        """Main chat function"""
        # Get relevant context
        relevant_chunks = self.get_relevant_chunks(query)

        # Generate and return response
        return self.generate_response(query, relevant_chunks)

def initialize_chatbot(pdf_path: str) -> PDFChatbot:
    """Initialize and prepare chatbot with a PDF"""
    chatbot = PDFChatbot()
    chunks = chatbot.preprocess_pdf(pdf_path)
    chatbot.create_embeddings()
    return chatbot

# Main execution
if __name__ == "__main__":
    # Initialize the chatbot with your PDF
    pdf_path = "Faiziab_Resume.pdf"
    chatbot = initialize_chatbot(pdf_path)

    # Ask questions
    response = chatbot.chat("What is the main topic of the document?")
    print(response)

    # Interactive chat loop
    print("\nEnter your questions (type 'quit' to exit):")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
        response = chatbot.chat(question)
        print("\nAnswer:", response)