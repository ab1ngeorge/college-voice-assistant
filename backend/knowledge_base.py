import json
import jsonlines
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

class KnowledgeBase:
    def __init__(self, db_path: str = "./chroma_db"):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection("college_knowledge")
        except:
            self.collection = self.chroma_client.create_collection(
                "college_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
    
    def load_documents(self, file_path: str = "documents.jsonl"):
        """Load documents from JSONL file into vector database"""
        documents = []
        embeddings = []
        ids = []
        
        with jsonlines.open(file_path) as reader:
            for i, doc in enumerate(reader):
                # Extract all language versions
                if isinstance(doc, dict) and 'text' in doc:
                    text_entry = doc['text']
                    # Split by pipe for different language versions
                    versions = [v.strip() for v in text_entry.split('|')]
                    
                    # Combine all versions for embedding
                    combined_text = ' '.join(versions)
                    
                    documents.append(combined_text)
                    ids.append(f"doc_{i}")
                    
                    # Store metadata with language versions
                    metadata = {
                        'english': versions[0] if len(versions) > 0 else '',
                        'malayalam': versions[1] if len(versions) > 1 else '',
                        'manglish': versions[2] if len(versions) > 2 else ''
                    }
                    
                    # Create embedding
                    embedding = self.embedding_model.encode(combined_text).tolist()
                    embeddings.append(embedding)
                    
                    # Add to collection
                    self.collection.add(
                        embeddings=[embedding],
                        documents=[combined_text],
                        metadatas=[metadata],
                        ids=[f"doc_{i}"]
                    )
        
        print(f"Loaded {len(documents)} documents into knowledge base")
        return len(documents)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        # Encode query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': results['distances'][0][i] if results['distances'] else 0
                })
        
        return formatted_results
    
    def add_document(self, english: str, malayalam: str, manglish: str = ""):
        """Add a new document to knowledge base"""
        combined_text = f"{english} | {malayalam}"
        if manglish:
            combined_text += f" | {manglish}"
        
        # Generate ID
        doc_id = f"doc_{int(np.random.rand() * 1000000)}"
        
        # Create embedding
        embedding = self.embedding_model.encode(combined_text).tolist()
        
        # Metadata
        metadata = {
            'english': english,
            'malayalam': malayalam,
            'manglish': manglish
        }
        
        # Add to collection
        self.collection.add(
            embeddings=[embedding],
            documents=[combined_text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return doc_id