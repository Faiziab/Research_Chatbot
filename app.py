import streamlit as st
from PDFChatbot import PDFChatbot, initialize_chatbot  # Import your PDFChatbot class

# Streamlit app interface
st.title("PDF Chatbot")

# Initialize a session state variable to keep track of chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
st.sidebar.header("Chat with PDF")
st.sidebar.write("Ask questions based on the content of the uploaded PDF.")

# User upload for PDF
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf is not None:
    # Save the uploaded file temporarily
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    # Reinitialize the chatbot with the newly uploaded PDF
    chatbot = initialize_chatbot("uploaded_pdf.pdf")
    st.success("PDF uploaded and processed successfully! You can now ask questions.")

    # Display past chat history
    for i, (user_message, bot_message) in enumerate(st.session_state.chat_history):
        st.write(f"**You:** {user_message}")
        st.write(f"**Bot:** {bot_message}")
        st.markdown("---")  # Add a separator between messages

    # User input for a new question
    user_query = st.text_input("Your Question", "")

    if user_query:
        with st.spinner("Processing..."):
            # Get the chatbot response for the user's query
            response = chatbot.chat(user_query)

        # Save the user query and bot response in the chat history
        st.session_state.chat_history.append((user_query, response))

        # Display the answer
        st.write("**Answer:**", response)

else:
    st.write("Please upload a PDF to get started.")
