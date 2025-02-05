import streamlit as st 
from ResearchChatbot import PDFChatbot
import os

def initialize_session_state():
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None

def main():
    st.set_page_config(page_title="Research Chatbot", layout="wide")
    
    # Initialize session state
    initialize_session_state()
    
    # Main layout
    st.title("Research PDF Chatbot")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        # Process the uploaded PDF
        if uploaded_file and (not st.session_state.chatbot or 
                            st.session_state.current_file != uploaded_file.name):
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Initialize chatbot and process PDF
            with st.spinner("Processing PDF..."):
                st.session_state.chatbot = PDFChatbot()
                st.session_state.chatbot.preprocess_pdf(temp_path)
                st.session_state.chatbot.create_embeddings()
                st.session_state.current_file = uploaded_file.name
            
            # Clean up temporary file
            os.remove(temp_path)
            st.success("PDF processed successfully!")

    # Main chat interface
    if st.session_state.chatbot:
        st.header(f"Chat about: {st.session_state.current_file}")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get and display chatbot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.chat(prompt)
                st.write(response)
            
            # Update chat history
            st.session_state.chat_history.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ])
    else:
        st.info("👈 Please upload a PDF file to start chatting.")

if __name__ == "__main__":
    main()