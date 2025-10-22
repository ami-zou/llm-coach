# embed_store.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
import re

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DB_DIR = "./chroma_store"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    # Split by pages first, then by sentences for better chunking
    pages = text.split("--- Page")
    chunks = []
    
    for page in pages:
        if not page.strip():
            continue
            
        # Clean up page content
        page_content = page.strip()
        if page_content.startswith("---"):
            page_content = page_content[3:].strip()
        
        # Split page into sentences
        sentences = re.split(r'[。！？]', page_content)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + sentence
            else:
                current_chunk += sentence + "。"
        
        # Add remaining content as final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    return chunks

def embed_and_store(filepath: str):
    """Embed and store text chunks in ChromaDB."""
    print(f"📖 Reading text from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    print("✂️ Chunking text...")
    chunks = chunk_text(text)
    print(f"📦 Created {len(chunks)} chunks")

    print("🤖 Loading embedding model...")
    embedder = SentenceTransformer(MODEL_NAME)
    
    print("🧮 Computing embeddings...")
    embeddings = embedder.encode(chunks)

    print("🗄️ Initializing ChromaDB...")
    # Use the new ChromaDB API
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name="notes",
        metadata={"hnsw:space": "cosine"}
    )

    print("💾 Storing chunks in ChromaDB...")
    # Prepare data for batch insertion
    documents = []
    ids = []
    embeddings_list = []
    metadatas = []
    
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        documents.append(chunk)
        ids.append(f"chunk-{i}")
        embeddings_list.append(emb.tolist())
        metadatas.append({
            "source": filepath,
            "chunk_id": i,
            "chunk_size": len(chunk)
        })
    
    # Batch insert all chunks
    collection.add(
        documents=documents,
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadatas
    )

    print(f"✅ Successfully stored {len(chunks)} chunks to ChromaDB at {DB_DIR}")
    print(f"📊 Collection info: {collection.count()} documents")

if __name__ == "__main__":
    embed_and_store("output_notes.txt")