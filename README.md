# 🩺 MedLocal-RAG: Privacy-First Medical Assistant

**MedLocal-RAG** is a Retrieval-Augmented Generation (RAG) chatbot designed to provide evidence-based medical information using only local resources. By leveraging quantized Large Language Models (LLMs) and local vector databases, this project ensures that sensitive queries never leave your machine, eliminating both data privacy concerns and cloud API costs.

---

## 🌟 Key Features

- **Total Privacy**: No data is sent to the cloud. All processing happens locally on your CPU/GPU.  
- **Medical Optimization**: Uses BioMistral, a model fine-tuned on PubMed research, for superior medical reasoning.  
- **Evidence-Based**: Every answer includes *"Clinical References"* pointing to specific pages in your local document library.  
- **Zero Cost**: Uses open-source tools (Ollama, LangChain, Streamlit) and free-to-run models.  
- **Simple UI**: A professional, clinical-themed dashboard for intuitive interaction.  

---

## 🏗️ Architecture

The system follows a standard RAG pipeline:

1. **Ingestion**: Medical PDFs are chunked and converted into vectors using the `all-minilm:l6-v2` model.  
2. **Storage**: Vectors are stored in a local ChromaDB instance.  
3. **Retrieval**: The system identifies the most relevant medical snippets for every user query.  
4. **Generation**: BioMistral synthesizes the snippets into a coherent, professional response.  

---

## 🛠️ Tech Stack

- **Orchestration**: LangChain  
- **Local Inference**: Ollama  
- **LLM**: BioMistral-7B (4-bit Quantized)  
- **Embeddings**: all-minilm:l6-v2 
- **Vector Store**: ChromaDB  
- **UI Framework**: Streamlit  

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+  
- Ollama installed on your system ([Download here](https://ollama.com))  
- At least 8GB RAM (16GB recommended)  

---

### 2. Install Local Models

Open your terminal and run:

```bash
ollama pull hf.co/MaziyarPanahi/BioMistral-7B-GGUF:Q4_K_M
ollama pull all-minilm:l6-v2
```

### 3. Setup the Project
Create environment:
```bash
# Conda
conda create -n medical-rag python=3.10
conda activate medical-rag
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Build Your Knowledge Base
- Add your medical PDFs or text files into 'data' folder.
- Run the ingestion script:
```bash
python ingest.py
```

### 5. Launch the Assistant
```bash
streamlit run app.py
```

The app will open automatically in your browser at:
👉 http://localhost:8501


## ⚠️ Disclaimer

MedLocal-RAG is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

---