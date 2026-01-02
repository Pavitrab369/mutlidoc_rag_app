import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_DIR = os.path.join(BASE_DIR, "docs_vectorstore")

embeddings = HuggingFaceEmbeddings()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0.0,
)

def process_document_to_chroma_db(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
    )

    texts = text_splitter.split_documents(documents)

    vectordb = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
    )

    vectordb.add_documents(texts)

    return 0

def answer_question(user_question):

    vectordb = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
    )

    retriever = vectordb.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer