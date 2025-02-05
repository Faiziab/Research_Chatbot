import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
import numpy as np
from typing import List, Dict
import re
import json
import os
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.style import Style
from rich.text import Text

class PDFChatbot:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", storage_dir: str = "pdf_storage"):
        self.console = Console()
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Loading models...", total=None)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        
        self.texts = []
        self.faiss_index = None
        self.current_pdf_id = None
        self.pdf_title = ""
    
    def preprocess_pdf(self, pdf_path: str, chunk_size: int = 1500, overlap: int = 200):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Processing PDF...", total=None)
            
            reader = PdfReader(pdf_path)
            raw_text = ""
            self.pdf_title = os.path.basename(pdf_path).rsplit('.', 1)[0]
            
            references_detected = False
            total_pages = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                progress.update(task, description=f"[cyan]Processing page {i+1}/{total_pages}")
                page_text = page.extract_text()
                processed_text = f"\n[Page {i+1}]\n"
                
                lines = page_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if re.match(r'^(REFERENCES|BIBLIOGRAPHY|WORKS CITED)$', line, re.IGNORECASE):
                        references_detected = True
                        break
                    
                    processed_text += line + "\n"
                
                if references_detected:
                    break
                
                raw_text += processed_text
            
            raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
            
            # Chunk processing
            progress.update(task, description="[cyan]Creating text chunks...")
            units = []
            current_unit = []
            current_length = 0
            
            for line in raw_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('[Page'):
                    if current_unit:
                        units.append(' '.join(current_unit))
                        current_unit = []
                        current_length = 0
                    current_unit.append(line)
                    continue
                
                if current_length + len(line) > chunk_size and current_unit:
                    units.append(' '.join(current_unit))
                    current_unit = []
                    current_length = 0
                
                current_unit.append(line)
                current_length += len(line)
            
            if current_unit:
                units.append(' '.join(current_unit))
            
            chunks = []
            for i in range(0, len(units)):
                chunk_text = units[i]
                next_idx = i + 1
                while (next_idx < len(units) and 
                       len(chunk_text) + len(units[next_idx]) < chunk_size + overlap):
                    chunk_text += ' ' + units[next_idx]
                    next_idx += 1
                chunks.append(chunk_text)
            
            self.texts = chunks
            self.store_pdf_data(pdf_path, chunks)
            
        return chunks

    def store_pdf_data(self, pdf_path: str, chunks: List[str]):
        storage_dir = "pdf_storage/data"
        os.makedirs(storage_dir, exist_ok=True)

        pdf_id = os.path.basename(pdf_path).rsplit('.', 1)[0]
        chunks_file = os.path.join(storage_dir, f"{pdf_id}.json")

        metadata = {
            "pdf_id": pdf_id,
            "title": self.pdf_title,
            "timestamp": datetime.now().isoformat()
        }

        data = {
            "chunks": chunks,
            "metadata": metadata
        }

        try:
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self.console.print(f"[green]Successfully saved extracted text to {chunks_file}")
        except Exception as e:
            self.console.print(f"[red]Error saving PDF data: {e}")

    def create_embeddings(self):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Creating embeddings...", total=None)
            embeddings = self.embed_model.encode(self.texts)
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(np.array(embeddings).astype('float32'))

    def get_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        query_embedding = self.embed_model.encode([query])
        distances, indices = self.faiss_index.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        relevant_chunks = []
        for i, dist in zip(indices[0], distances[0]):
            if dist < 20:
                relevant_chunks.append(self.texts[i])
        
        return relevant_chunks

    def generate_response(self, query: str, context: List[str]) -> str:
        prompt = f"""You are a helpful AI assistant that answers questions about the document \"{self.pdf_title}\". 
Your task is to answer questions using ONLY the information provided in the following context.
If the context doesn't contain enough information to answer the question, say "I cannot answer this question based on the provided context." 
Do not make up or infer information that is not directly supported by the context.

Context from the document:
{' '.join(context)}

Question: {query}

Provide a precise answer based solely on the above context. Include relevant page numbers if available.

Answer:"""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Generating response...", total=None)
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=2048
            ).to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Answer:")[-1].strip()
        
        if not context:
            return "I cannot find relevant information in the document to answer this question."
            
        return answer

    def chat(self, query: str) -> str:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Searching document...", total=None)
            relevant_chunks = self.get_relevant_chunks(query)
            
        if not relevant_chunks:
            return "I cannot find relevant information in the document to answer this question."
        
        return self.generate_response(query, relevant_chunks)

def display_welcome():
    console = Console()
    welcome_text = """
# PDF Chatbot

Welcome to the interactive PDF Chatbot! This tool allows you to:
- Load and process PDF documents
- Ask questions about the content
- Get AI-powered responses based on the document

Type 'quit' at any time to exit the program.
    """
    console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="cyan"))

def main():
    console = Console()
    display_welcome()
    
    # Get PDF path
    pdf_path = Prompt.ask("[cyan]Enter the path to your PDF file")
    
    try:
        chatbot = PDFChatbot()
        chatbot.preprocess_pdf(pdf_path)
        chatbot.create_embeddings()
        
        console.print("\n[green]✓ PDF processed and ready for questions!")
        
        # Create a table to show document info
        table = Table(title="Document Information", border_style="cyan")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Document", os.path.basename(pdf_path))
        table.add_row("Status", "Ready")
        console.print(table)
        
        while True:
            # Create an input panel
            question = Prompt.ask("\n[cyan]Your question", 
                                default="quit",
                                show_default=False)
            
            if question.lower() == 'quit':
                if Confirm.ask("[yellow]Are you sure you want to quit?"):
                    break
                continue
            
            # Display the response in a panel
            response = chatbot.chat(question)
            console.print(Panel(response, title="Answer", border_style="green"))
        
        console.print("\n[cyan]Thank you for using the PDF Chatbot! Goodbye! 👋")
        
    except Exception as e:
        console.print(f"\n[red]An error occurred: {str(e)}")
        console.print("[yellow]Please try again with a valid PDF file.")

if __name__ == "__main__":
    main()