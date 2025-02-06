import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
import numpy as np
import re
import json
import os
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown


class PDFChatbot:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.console = Console()
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # Force CPU usage

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            progress.add_task("[cyan]Loading models...", total=None)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # CPU-friendly format
                device_map=None,  # Forces CPU usage
                low_cpu_mem_usage=True  
            )

        self.texts = []
        self.faiss_index = None
        self.pdf_title = ""

    def preprocess_pdf(self, pdf_path: str, chunk_size: int = 1500, overlap: int = 200):
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            task = progress.add_task("[cyan]Processing PDF...", total=None)

            reader = PdfReader(pdf_path)
            raw_text = ""
            self.pdf_title = os.path.basename(pdf_path).rsplit(".", 1)[0]
            references_detected = False

            for i, page in enumerate(reader.pages):
                progress.update(task, description=f"[cyan]Processing page {i+1}/{len(reader.pages)}")
                page_text = page.extract_text()
                processed_text = f"\n[Page {i+1}]\n"

                for line in page_text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if re.match(r"^(REFERENCES|BIBLIOGRAPHY|WORKS CITED)$", line, re.IGNORECASE):
                        references_detected = True
                        break
                    processed_text += line + "\n"

                if references_detected:
                    break

                raw_text += processed_text

            raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
            self.texts = self.chunk_text(raw_text, chunk_size, overlap)
            self.store_pdf_data(pdf_path, self.texts)

        return self.texts

    def chunk_text(self, text, chunk_size, overlap):
        units = []
        current_unit = []
        current_length = 0

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("[Page"):
                if current_unit:
                    units.append(" ".join(current_unit))
                    current_unit = []
                    current_length = 0
                current_unit.append(line)
                continue

            if current_length + len(line) > chunk_size and current_unit:
                units.append(" ".join(current_unit))
                current_unit = []
                current_length = 0

            current_unit.append(line)
            current_length += len(line)

        if current_unit:
            units.append(" ".join(current_unit))

        chunks = []
        for i in range(len(units)):
            chunk_text = units[i]
            next_idx = i + 1
            while next_idx < len(units) and len(chunk_text) + len(units[next_idx]) < chunk_size + overlap:
                chunk_text += " " + units[next_idx]
                next_idx += 1
            chunks.append(chunk_text)

        return chunks

    def store_pdf_data(self, pdf_path, chunks):
        storage_dir = "pdf_storage/data"
        os.makedirs(storage_dir, exist_ok=True)

        pdf_id = os.path.basename(pdf_path).rsplit(".", 1)[0]
        chunks_file = os.path.join(storage_dir, f"{pdf_id}.json")

        metadata = {
            "pdf_id": pdf_id,
            "title": self.pdf_title,
            "timestamp": datetime.now().isoformat(),
        }

        data = {"chunks": chunks, "metadata": metadata}

        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.console.print(f"[green]Successfully saved extracted text to {chunks_file}")

    def create_embeddings(self):
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            task = progress.add_task("[cyan]Creating embeddings...", total=None)
            embeddings = self.embed_model.encode(self.texts)
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(np.array(embeddings).astype("float32"))

    def get_relevant_chunks(self, query, k=3):
        query_embedding = self.embed_model.encode([query])
        distances, indices = self.faiss_index.search(np.array(query_embedding).astype("float32"), k)
        return [self.texts[i] for i, dist in zip(indices[0], distances[0]) if dist < 20]

    def generate_response(self, query, context):
        prompt = f"""You are a helpful AI assistant answering questions about \"{self.pdf_title}\".
Use only the information provided below. If there's not enough context, say so.

Context:
{" ".join(context)}

Question: {query}

Answer:"""

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            task = progress.add_task("[cyan]Generating response...", total=None)

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=2048).to("cpu")

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,  # Reduced for better performance on CPU
                repetition_penalty=1.1,
                do_sample=False,
                num_beams=3,  # Lower beam count for faster inference
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()

    def chat(self, query):
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            task = progress.add_task("[cyan]Searching document...", total=None)
            relevant_chunks = self.get_relevant_chunks(query)

        if not relevant_chunks:
            return "I cannot find relevant information in the document to answer this question."

        return self.generate_response(query, relevant_chunks)


def display_welcome():
    console = Console()
    welcome_text = """
# PDF Chatbot

Welcome! This tool allows you to:
- Load and process PDFs
- Ask questions and get AI-powered answers

Type 'quit' to exit.
    """
    console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="cyan"))


def main():
    console = Console()
    display_welcome()

    pdf_path = Prompt.ask("[cyan]Enter the path to your PDF file")
    chatbot = PDFChatbot()
    chatbot.preprocess_pdf(pdf_path)
    chatbot.create_embeddings()

    while True:
        question = Prompt.ask("\n[cyan]Your question", default="quit", show_default=False)
        if question.lower() == "quit" and Confirm.ask("[yellow]Are you sure?"):
            break
        console.print(Panel(chatbot.chat(question), title="Answer", border_style="green"))


if __name__ == "__main__":
    main()
