import os
import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

st.set_page_config(page_title="MedLocal-RAG Assistant", layout="wide")
st.title("🩺 MedLocal-RAG Assistant")
st.markdown("---")

# Initialize Local Models
llm = OllamaLLM(model="cniongolo/biomistral")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load existing Vector DB
if os.path.exists("./vector_db"):
    vector_db = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Setup RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
else:
    st.error("Vector database not found. Please run 'python ingest.py' first.")
    st.stop()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = qa_chain.invoke({"query": prompt})
        answer = response["result"]
        sources = response["source_documents"]
        
        st.markdown(answer)
        with st.expander("View Evidence Sources"):
            for doc in sources:
                st.write(f"- {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})