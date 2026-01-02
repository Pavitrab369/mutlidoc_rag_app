import os
import streamlit as st
from rag_utility import process_document_to_chroma_db, answer_question

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

st.title("llama-3.3-70B Versatile - Document RAG")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader("Choose a file", type="pdf",accept_multiple_files=True)

if uploaded_files is not None:
    print(uploaded_files)
    for file in uploaded_files:
        save_path = os.path.join(DATA_DIR, file.name)

        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

        process_document = process_document_to_chroma_db(save_path)
    st.info("Document Processed Successfully!")

user_question = st.text_input("Ask your question about the document")

if st.button("answer"):

    answer = answer_question(user_question)
    st.markdown("### llama-3.3-70B Versatile's Answer:")
    st.markdown(answer)
