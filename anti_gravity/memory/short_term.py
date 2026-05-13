"""
Short-term memory using LangGraph's checkpointing
Stores conversation state within a session
"""

import pickle
from typing import Dict, Any, Optional
from datetime import datetime
import os

# Simple file-based checkpointer (will upgrade to PostgreSQL later)
class FileCheckpointer:
    """Simple file-based checkpointer for short-term memory"""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, thread_id: str, state: Dict[str, Any]) -> str:
        """Save state for a thread"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{thread_id}.pkl")
        data = {
            "state": state,
            "timestamp": datetime.now().isoformat(),
            "thread_id": thread_id
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        return checkpoint_path
    
    def load(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Load state for a thread"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{thread_id}.pkl")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            return data["state"]
        return None
    
    def list_threads(self) -> list:
        """List all available threads"""
        threads = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith(".pkl"):
                threads.append(file.replace(".pkl", ""))
        return threads

# Global checkpointer instance
checkpointer = FileCheckpointer()

def get_short_term_memory(thread_id: str) -> Optional[Dict]:
    """Retrieve short-term memory for a thread"""
    return checkpointer.load(thread_id)

def save_short_term_memory(thread_id: str, state: Dict) -> None:
    """Save short-term memory for a thread"""
    checkpointer.save(thread_id, state)
