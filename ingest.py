import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def build_vector_db():
    # 1. Load Documents
    loader = PyPDFDirectoryLoader("./data")
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    # 2. Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # 3. Initialize Embedding Model (Nomic-Embed is high-performing and local)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Create and Persist Vector Database
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./vector_db"
    )
    print("Vector database created and saved to ./vector_db")

if __name__ == "__main__":
    build_vector_db()