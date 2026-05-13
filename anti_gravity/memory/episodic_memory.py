"""
Episodic memory using SQLite (PostgreSQL later)
Stores complete history of runs with metrics
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import os

class EpisodicMemory:
    """Stores complete history of runs (episodes)"""
    
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create episodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT UNIQUE,
                user_id TEXT,
                timestamp TEXT,
                task TEXT,
                dataset_name TEXT,
                dataset_shape TEXT,
                model_metrics TEXT,
                artifacts TEXT,
                duration_seconds REAL,
                success BOOLEAN,
                error_message TEXT
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_id ON episodes(user_id)
        ''')
        cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_timestamp
    ON episodes(timestamp)
''')
        
        conn.commit()
        conn.close()
        print(f"   📀 Episodic memory initialized at {self.db_path}")
    
    def save_episode(self, user_id: str, task: str, state: Dict[str, Any], 
                     duration: float, success: bool, error: str = None) -> str:
        """Save a complete run episode"""
        import uuid
        episode_id = str(uuid.uuid4())[:8]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract info from state
        dataset_info = state.get('dataset_info', {})
        dataset_name = state.get('raw_data_path', 'unknown')
        
        cursor.execute('''
            INSERT INTO episodes 
            (episode_id, user_id, timestamp, task, dataset_name, dataset_shape, 
             model_metrics, artifacts, duration_seconds, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            episode_id,
            user_id,
            datetime.now().isoformat(),
            task,
            dataset_name,
            json.dumps(dataset_info.get('shape', [0, 0])),
            json.dumps(state.get('model_metrics', {})),
            json.dumps(state.get('artifacts', {})),
            duration,
            success,
            error
        ))
        
        conn.commit()
        conn.close()
        
        print(f"   📝 Saved episode {episode_id} for user {user_id}")
        return episode_id
    
    def get_recent_episodes(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get recent episodes for a user"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM episodes 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        episodes = []
        for row in cursor.fetchall():
            episode = dict(row)
            # Parse JSON fields
            episode['dataset_shape'] = json.loads(episode['dataset_shape']) if episode['dataset_shape'] else []
            episode['model_metrics'] = json.loads(episode['model_metrics']) if episode['model_metrics'] else {}
            episode['artifacts'] = json.loads(episode['artifacts']) if episode['artifacts'] else {}
            episodes.append(episode)
        
        conn.close()
        return episodes
    
    def get_episode_stats(self, user_id: str) -> Dict:
        """Get statistics about user's episodes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_runs,
                AVG(CASE WHEN success THEN 1 ELSE 0 END) as success_rate,
                AVG(duration_seconds) as avg_duration,
                COUNT(DISTINCT task) as unique_tasks
            FROM episodes 
            WHERE user_id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            "total_runs": row[0] or 0,
            "success_rate": row[1] or 0,
            "avg_duration": row[2] or 0,
            "unique_tasks": row[3] or 0
        }

# Global instance
episodic_memory = EpisodicMemory()
