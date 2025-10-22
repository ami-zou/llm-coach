# Local RAG Chatbot with Deepseek + Ollama + Chroma

This project lets you run a **local Retrieval-Augmented Generation (RAG)** chatbot using **Deepseek or Ollama models**, powered by **ChromaDB** as a lightweight vector store. It's optimized for macOS M1/M2/M4 chips but can work elsewhere with minimal changes.

---

## 📦 Project Structure

| File | Purpose |
|------|---------|
| `data_processing.py` | Clean and prepare your raw `.txt` or `.md` files |
| `embed_store.py` | Embed your documents and store them in Chroma vector DB |
| `query_rag.py` | Query the stored knowledge base using a local LLM |
| `run_deepseek.py` | Run Deepseek 7B-Instruct using HuggingFace Transformers |
| `run_ollama_rag.py` | Run Ollama RAG with any local Ollama model |
| `setup_ollama.py` | Install and pull models via Ollama CLI |

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
### 2. Prepare Your Data

Put your source documents (.txt, .md, etc.) in a folder, then run:
```bash
python data_processing.py --source_dir ./your_data
```
### 3. Embed Documents
```bash
python embed_store.py
```
This creates a chroma_store folder with your vectorized knowledge base.

### 4. Run a Local Model

Choose either:

#### Deepseek (via HuggingFace):
```bash
python run_deepseek.py
```
#### Ollama (must have ollama installed):
```bash
# Start Ollama in background
ollama run llama3

# Then run
python run_ollama_rag.py
```
### 5. Ask Questions

Once a model is running, run:
```bash
python query_rag.py
```
Enter questions in the terminal and get local answers from your own docs!
