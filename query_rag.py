# query_rag.py
import chromadb
from sentence_transformers import SentenceTransformer
import os

DB_DIR = "./chroma_store"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def query_notes(query: str, n_results: int = 5):
    """Query the ChromaDB for relevant notes."""
    print(f"🔍 Querying: '{query}'")
    
    # Load the embedding model
    embedder = SentenceTransformer(MODEL_NAME)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection("notes")
    
    # Generate query embedding
    query_embedding = embedder.encode([query])
    
    # Search for similar documents
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    
    print(f"📊 Found {len(results['documents'][0])} relevant chunks:")
    print("=" * 80)
    
    for i, (doc, distance, metadata) in enumerate(zip(
        results['documents'][0], 
        results['distances'][0], 
        results['metadatas'][0]
    )):
        print(f"\n📄 Result {i+1} (Similarity: {1-distance:.3f})")
        print(f"📏 Chunk size: {metadata['chunk_size']} characters")
        print(f"📝 Content: {doc[:200]}{'...' if len(doc) > 200 else ''}")
        print("-" * 40)
    
    return results

def retrieve_relevant_chunks(query: str, n_results: int = 5):
    """Retrieve relevant chunks for RAG without printing details."""
    # Load the embedding model
    embedder = SentenceTransformer(MODEL_NAME)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection("notes")
    
    # Generate query embedding
    query_embedding = embedder.encode([query])
    
    # Search for similar documents
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    
    # Return just the document chunks
    return results['documents'][0]

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "翻盘",
        "价值交换"
    ]
    
    for query in test_queries:
        query_notes(query)
        print("\n" + "="*80 + "\n")
