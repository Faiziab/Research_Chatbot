# main.py
import os
from pdf_utils import extract_text_from_pdf
from faiss_utils import chunk_text, create_faiss_index, retrieve_top_k_passages
from model_utils import load_model, generate_answer
from tqdm import tqdm

def answer_question(pdf_path, user_query, model, tokenizer):
    """
    This function processes a question, finds the most relevant passages from the PDF,
    and generates a detailed answer with progress bars.
    """
    # Step 1: Extract text from PDF
    print("\nProcessing your question...")
    print("1. Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Chunk the extracted text
    print("2. Chunking text...")
    chunks = chunk_text(pdf_text)
    
    # Step 3: Create FAISS index
    print("3. Creating search index...")
    index = create_faiss_index(chunks)
    
    # Step 4: Retrieve relevant passages
    print("4. Finding relevant passages...")
    top_passages = retrieve_top_k_passages(user_query, index, chunks)
    
    # Step 5: Generate answer
    print("5. Generating detailed answer...(this could take a while)")
    context = """You are an advanced AI model tasked with analyzing and understanding a provided PDF document. Your primary responsibility is to extract factual and contextually relevant information from the document and respond to user queries accordingly.

Capabilities:
- **Document Understanding:** You are capable of comprehending the structure, key points, numerical data, tables, and other significant elements of the document.
- **Contextual Relevance:** Ensure that all responses are directly related to the content of the PDF, reflecting the information accurately and avoiding irrelevant details.
- **Data Extraction & Integrity:** When answering questions about specific statistics or figures, always extract the exact values from the document (if available). If such data is missing or unclear, state that explicitly.
- **Summarization & Analysis:** You can summarize sections, compare details, and generate insights based on the content of the document.
- **Formatting:** When providing extracted data, especially numerical statistics or tables, ensure clarity and proper structure.

Behavior Guidelines:
1. **Fact-Checking:** If a question requires specific numbers, dates, or statistics, ensure the information is directly extracted from the PDF. Do not guess or approximate.
2. **Contextual Awareness:** Ensure responses align with the wording, tone, and details of the document. Avoid speculating or adding unverified information.
3. **Citations:** When relevant, refer to the section or page number where the information is found. For example, “According to Section 3.2 on page 12...”
4. **Handling Ambiguity:** If the requested information is not found in the document, state so clearly. For example, “The document does not mention specific figures regarding X, but it discusses related details in Section Y.”
5. **Extracting Structured Data:** When the document contains tables, lists, or bullet points, format the extracted data clearly and concisely. 
 """.join(top_passages)
    answer = generate_answer(user_query, context, model, tokenizer)
    
    return answer

def interactive_session(pdf_path):
    """
    This function allows continuous user interaction until 'quit' is typed.
    """
    print("Initializing...")
    print("Loading model (this might take a few moments)...")
    model, tokenizer = load_model()
    print("Model loaded successfully!")

    while True:
        user_query = input("\nEnter your question (or type 'quit' to exit): ")
        
        if user_query.lower() == 'quit':
            print("Exiting the chatbot. Goodbye!")
            break
        
        answer = answer_question(pdf_path, user_query, model, tokenizer)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    pdf_path = "research_paper.pdf"
    interactive_session(pdf_path)