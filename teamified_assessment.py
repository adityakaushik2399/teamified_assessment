import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any
import requests
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from io import BytesIO
import re
import pickle

class PhilippineHistoryRAG:
    def __init__(self, 
                 pdf_path: str = "PHILIPPINE-HISTORY-SOURCE-BOOK-FINAL-SEP022021.pdf",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 ollama_model: str = "gemma3:1b",
                 ollama_url: str = "http://localhost:11434",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize the Philippine History RAG system.
        
        Args:
            pdf_path: Path to the Philippine history PDF file
            embedding_model: Name of the sentence transformer model for embeddings
            ollama_model: Name of the Ollama model to use
            ollama_url: Base URL for Ollama API
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.pdf_path = pdf_path
        self.embedding_model_name = embedding_model
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Storage for chunks and index
        self.chunks = []
        self.faiss_index = None
        
    def load_and_extract_text(self) -> str:
        """
        Load the PDF file and extract text content using PyMuPDF.
        
        Returns:
            Extracted text from the PDF
        """
        print(f"Loading PDF: {self.pdf_path}")
        
        try:
            # Open the PDF document
            pdf_document = fitz.open(self.pdf_path)
            text = ""
            
            print(f"Processing {len(pdf_document)} pages...")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                print(f"Extracted page {page_num + 1}/{len(pdf_document)}")
            
            # Close the document
            pdf_document.close()
            
            print(f"Successfully extracted {len(text)} characters from PDF")
            return text
                
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess the extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'[.,;:!?]{2,}', '.', text)
        
        return text.strip()
    
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        print("Creating text chunks...")
        
        cleaned_text = self.clean_text(text)
        chunks = []
        
        # Split by sentences first for better chunk boundaries
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'length': len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'length': len(current_chunk)
            })
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create embeddings for all chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        print("Creating embeddings...")
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            FAISS index
        """
        print("Building FAISS index...")
        
        # Create a FAISS index (using L2 distance)
        index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add embeddings to index
        index.add(embeddings.astype(np.float32))
        
        print(f"FAISS index built with {index.ntotal} vectors")
        return index
    
    def setup_index(self):
        """
        Complete setup: load PDF, create chunks, embeddings, and FAISS index.
        """
        print("=== Setting up Philippine History RAG System ===")
        
        # Load and process PDF
        text = self.load_and_extract_text()
        self.chunks = self.create_chunks(text)
        
        # Create embeddings and FAISS index
        embeddings = self.create_embeddings(self.chunks)
        self.faiss_index = self.build_faiss_index(embeddings)
        
        print("=== Setup complete! ===\n")
    
    def save_index(self, index_path: str = "faiss_index.index", chunks_path: str = "chunks.pkl"):
        """
        Save the FAISS index and chunks to disk.
        
        Args:
            index_path: Path to save FAISS index
            chunks_path: Path to save chunks
        """
        if self.faiss_index is None:
            raise ValueError("No index to save. Run setup_index() first.")
            
        faiss.write_index(self.faiss_index, index_path)
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"Index saved to {index_path}, chunks saved to {chunks_path}")
    
    def load_index(self, index_path: str = "faiss_index.index", chunks_path: str = "chunks.pkl"):
        """
        Load the FAISS index and chunks from disk.
        
        Args:
            index_path: Path to load FAISS index
            chunks_path: Path to load chunks
        """
        self.faiss_index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"Index loaded from {index_path}, chunks loaded from {chunks_path}")
    
    def query_ollama(self, prompt: str) -> str:
        """
        Send a query to Ollama and get response.
        
        Args:
            prompt: The prompt to send to Ollama
            
        Returns:
            Response from Ollama
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Make sure Ollama is running on localhost:11434"
        except requests.exceptions.Timeout:
            return "Error: Request to Ollama timed out"
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if self.faiss_index is None:
            raise ValueError("Index not built. Run setup_index() first.")
        
        # Create embedding for the query
        query_embedding = self.embedding_model.encode([query])
        
        # Search the index
        distances, indices = self.faiss_index.search(query_embedding.astype(np.float32), top_k)
        
        # Get relevant chunks
        relevant_chunks = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(distance)
                chunk['rank'] = i + 1
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def build_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from relevant chunks.
        
        Args:
            relevant_chunks: List of relevant chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"[Chunk {chunk['rank']}]: {chunk['text']}")
        
        return "\n\n".join(context_parts)
    
    def answer_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a query using the RAG system.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary containing the answer and metadata
        """
        print(f"Processing query: {query}")
        print("-" * 50)
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k)
        
        if not relevant_chunks:
            return {
                'query': query,
                'answer': "I couldn't find relevant information to answer your question.",
                'relevant_chunks': [],
                'context': ""
            }
        
        # Build context
        context = self.build_context(relevant_chunks)
        
        # Create prompt for Ollama
        prompt = f"""Based on the following context about Philippine history, please answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer: """
        
        print("Retrieving answer from Ollama...")
        answer = self.query_ollama(prompt)
        
        return {
            'query': query,
            'answer': answer,
            'relevant_chunks': relevant_chunks,
            'context': context
        }
    
    def print_detailed_response(self, result: Dict[str, Any]):
        """
        Print a detailed response with context and sources.
        
        Args:
            result: Result dictionary from answer_query
        """
        print("=" * 60)
        print("PHILIPPINE HISTORY RAG SYSTEM - QUERY RESULT")
        print("=" * 60)
        print(f"Query: {result['query']}")
        print("-" * 60)
        print("ANSWER:")
        print(result['answer'])
        print("-" * 60)
        print(f"RETRIEVED CONTEXT ({len(result['relevant_chunks'])} chunks):")
        
        for chunk in result['relevant_chunks']:
            print(f"\n[Chunk {chunk['rank']} - Score: {chunk['similarity_score']:.4f}]")
            print(chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'])
        
        print("=" * 60)

def main():
    """
    Main function to demonstrate the RAG system.
    """
    # Initialize the RAG system
    rag = PhilippineHistoryRAG(
        pdf_path="PHILIPPINE-HISTORY-SOURCE-BOOK-FINAL-SEP022021.pdf",
        embedding_model="all-MiniLM-L6-v2",
        ollama_model="gemma3:1b",  # Change this to your preferred Ollama model
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Check if saved index exists
    if os.path.exists("faiss_index.index") and os.path.exists("chunks.pkl"):
        print("Found existing index files. Loading...")
        rag.load_index()
    else:
        print("No existing index found. Creating new index...")
        rag.setup_index()
        # Save for future use
        rag.save_index()
    
    # Example queries
    sample_queries = [
        "When did the EDSA People Power Revolution happen?",
        "Who was Jose Rizal?",
        "What happened during the Spanish colonization of the Philippines?",
        "Tell me about Ferdinand Marcos and Martial Law",
        "What was the Katipunan?"
    ]
    
    print("Philippine History RAG System is ready!")
    print("\nSample queries you can try:")
    for i, query in enumerate(sample_queries, 1):
        print(f"{i}. {query}")
    
    # Interactive query loop
    while True:
        print("\n" + "="*50)
        user_query = input("Enter your question about Philippine history (or 'quit' to exit): ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the Philippine History RAG system!")
            break
        
        if not user_query:
            print("Please enter a valid question.")
            continue
        
        try:
            # Get answer
            result = rag.answer_query(user_query, top_k=3)
            
            # Print detailed response
            rag.print_detailed_response(result)
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()