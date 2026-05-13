"""
Memory Agent - Integrates all memory systems
Learns from past runs and improves future decisions
"""

import time
from anti_gravity.memory.semantic_memory import semantic_memory
from anti_gravity.memory.episodic_memory import episodic_memory
from anti_gravity.memory.short_term import get_short_term_memory, save_short_term_memory
from anti_gravity.state import AntiGravityState

class MemoryAgent:
    """Coordinates all memory systems"""
    
    def __init__(self):
        self.start_time = None
    
    def before_pipeline(self, state: AntiGravityState) -> dict:
        """Called before pipeline starts - retrieve relevant memories"""
        print("🧠 Memory Agent: Retrieving relevant past experiences...")
        
        user_id = state["user_id"]
        task = state["current_task"]
        
        # Get short-term memory (current session)
        short_term = get_short_term_memory(f"{user_id}_{task}")
        if short_term:
            print(f"   📖 Found short-term memory from this session")
        
        # Find similar past tasks using semantic search
        similar = semantic_memory.find_similar(
            user_id=user_id,
            query=f"Performing {task} on dataset {state.get('raw_data_path', 'unknown')}",
            n_results=2
        )
        
        if similar:
            print(f"   🔍 Found {len(similar)} similar past experiences")
            for mem in similar:
                preview = mem['metadata'].get('result_preview', '')[:100]
                print(f"      - Similarity: {1 - mem['similarity_score']:.2f} | {preview}")
        
        # Get episode statistics
        stats = episodic_memory.get_episode_stats(user_id)
        if stats['total_runs'] > 0:
            print(f"   📊 User stats: {stats['total_runs']} runs, "
                  f"{stats['success_rate']*100:.0f}% success rate")
        
        # Add memory context to state
        memory_context = {
            "similar_past_tasks": similar,
            "user_stats": stats,
            "has_short_term": short_term is not None
        }
        
        state["messages"].append({
            "role": "memory", 
            "content": f"Retrieved {len(similar)} similar experiences"
        })
        
        return {"messages": state["messages"]}
    
    def after_pipeline(self, state: AntiGravityState, success: bool, error: str = None) -> dict:
        """Called after pipeline completes - store memories"""
        print("🧠 Memory Agent: Storing experiences...")
        
        user_id = state["user_id"]
        task = state["current_task"]
        duration = time.time() - self.start_time if self.start_time else 0
        
        # Save short-term memory
        save_short_term_memory(f"{user_id}_{task}", state)
        
        # Save episodic memory (complete run)
        episode_id = episodic_memory.save_episode(
            user_id=user_id,
            task=task,
            state=state,
            duration=duration,
            success=success,
            error=error
        )
        
        # Save semantic memory (for similarity search)
        if success:
            result_summary = {
                "accuracy": state.get("model_metrics", {}).get("accuracy", 
                           state.get("model_metrics", {}).get("r2_score", "N/A")),
                "dataset_shape": state.get("dataset_info", {}).get("shape", "N/A")
            }
            semantic_memory.add_memory(
                user_id=user_id,
                task=task,
                result=result_summary
            )
        
        print(f"   ✅ Memories stored (Episode: {episode_id})")
        
        return {"messages": state["messages"]}
    
    def start_timer(self):
        """Start timing the pipeline"""
        self.start_time = time.time()

# Global instance
memory_agent = MemoryAgent()
