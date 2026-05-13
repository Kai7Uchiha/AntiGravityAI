"""
Semantic memory using ChromaDB for similarity search
Stores embeddings of past analyses for retrieval
"""

import chromadb
from chromadb.utils import embedding_functions
import hashlib
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

class SemanticMemory:
    """Vector database for semantic similarity search"""
    
    def __init__(self, collection_name="anti_gravity_memory"):
        # Use local persistent client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Use sentence transformers for embeddings (local, free)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
        
        print(f"   🧠 Semantic memory initialized with {self.collection.count()} existing memories")
    
    def add_memory(self, user_id: str, task: str, result: Dict, embedding_text: str = None):
        """Store a memory with embedding for later retrieval"""
        
        # Create unique ID based on content
        memory_id = hashlib.md5(f"{user_id}_{task}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        # Create text to embed (if not provided)
        if embedding_text is None:
            embedding_text = f"User {user_id} performed task: {task}. Result: {json.dumps(result, default=str)}"
        
        # Metadata for filtering
        metadata = {
            "user_id": user_id,
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "result_preview": json.dumps(result, default=str)[:200]
        }
        
        # Add to collection
        self.collection.add(
            ids=[memory_id],
            metadatas=[metadata],
            documents=[embedding_text]
        )
        
        print(f"   💾 Saved semantic memory: {task[:50]}...")
        return memory_id
    
    def find_similar(self, user_id: str, query: str, n_results: int = 3) -> List[Dict]:
        """Find similar past memories based on semantic similarity"""
        
        # Search only for this user's memories
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"user_id": user_id}
        )
        
        similar_memories = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                memory = {
                    "id": results['ids'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "similarity_score": results['distances'][0][i] if results['distances'] else None,
                    "document": results['documents'][0][i] if results['documents'] else None
                }
                similar_memories.append(memory)
        
        return similar_memories
    
    def get_user_memories(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get all memories for a specific user"""
        # Get all memories for user
        results = self.collection.get(
            where={"user_id": user_id},
            limit=limit
        )
        
        memories = []
        if results['ids']:
            for i in range(len(results['ids'])):
                memory = {
                    "id": results['ids'][i],
                    "metadata": results['metadatas'][i] if results['metadatas'] else {},
                    "document": results['documents'][i] if results['documents'] else None
                }
                memories.append(memory)
        
        return memories
    
    def clear_user_memories(self, user_id: str):
        """Clear all memories for a user (for testing)"""
        # Get all IDs for this user
        results = self.collection.get(where={"user_id": user_id})
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"   🗑️  Cleared {len(results['ids'])} memories for user {user_id}")

# Global instance
semantic_memory = SemanticMemory()
