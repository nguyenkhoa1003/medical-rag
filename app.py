import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

st.set_page_config(page_title="MedLocal-RAG Assistant", page_icon="🩺", layout="wide")

# Custom CSS for Premium UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1f1c2c 0%, #928dab 100%);
        color: white;
    }
    
    .stChatInputContainer {
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Modify primary color */
    :root {
        --primary-color: #6a11cb;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.title("🩺 MedLocal-RAG")
    st.markdown("Your locally hosted AI medical assistant.")
    st.markdown("---")
    st.markdown("**Model:** MaziyarPanahi/BioMistral-7B-GGUF")
    st.markdown("**Embeddings:** all-minilm:l6-v2")
    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("🩺 MedLocal-RAG Assistant")
st.markdown("Welcome to your intelligent, locally-hosted medical RAG system. Feel free to ask any medical queries based on the ingested documents!")

# from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_ollama import OllamaLLM, OllamaEmbeddings
try:
    llm = OllamaLLM(model="hf.co/MaziyarPanahi/BioMistral-7B-GGUF:Q4_K_M")
    embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")
except Exception as e:
    st.error(f"Error loading models: {e}")

# Load existing Vector DB
if os.path.exists("./vector_db"):
    try:
        vector_db = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        
        # Setup RAG Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error loading vector db: {e}")
        st.stop()
else:
    st.warning("⚠️ Vector database not found. Please run 'python ingest.py' first.")
    st.stop()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    # Use different avatars
    avatar = "🧑‍⚕️" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a medical question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        with st.spinner("Retrieving local context and generating answer..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]
            except Exception as e:
                answer = f"Sorry, an error occurred during processing: {str(e)}"
                sources = []
                
        st.markdown(answer)
        if sources:
            with st.expander("📚 View Evidence Sources"):
                for doc in sources:
                    source_name = doc.metadata.get('source', 'Unknown')
                    page_num = doc.metadata.get('page', 'N/A')
                    st.markdown(f"- **{source_name}** (Page {page_num})")
                    st.caption(f"_{doc.page_content[:200]}..._")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})