from langgraph.graph import StateGraph, END
import pandas as pd
import hashlib
from datetime import datetime
from anti_gravity.state import AntiGravityState
from anti_gravity.agents.quality_agent import data_quality_agent
from anti_gravity.agents.explorer_agent import explorer_agent
from anti_gravity.agents.trainer_agent import model_trainer_agent
from anti_gravity.memory_agent import memory_agent

def create_workflow():
    workflow = StateGraph(AntiGravityState)
    
    workflow.add_node("memory_before", lambda state: memory_agent.before_pipeline(state))
    workflow.add_node("data_quality", data_quality_agent)
    workflow.add_node("explorer", explorer_agent)
    workflow.add_node("trainer", model_trainer_agent)
    workflow.add_node("memory_after", lambda state: {})
    
    workflow.set_entry_point("memory_before")
    workflow.add_edge("memory_before", "data_quality")
    workflow.add_edge("data_quality", "explorer")
    workflow.add_edge("explorer", "trainer")
    workflow.add_edge("trainer", "memory_after")
    workflow.add_edge("memory_after", END)
    
    return workflow.compile()

def run_full_pipeline(raw_data_path: str, user_id: str = "default_user"):
    """Run pipeline with dataframe loaded once"""
    
    # Generate run ID
    run_id = hashlib.sha256(
        f"{user_id}_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]
    
    # Load dataframe ONCE
    print(f"📖 Loading dataset once (run_id: {run_id})...")
    df = pd.read_csv(raw_data_path)
    
    initial_state = AntiGravityState(
        messages=[],
        dataset_info={},
        artifacts={},
        user_id=user_id,
        current_task="full_pipeline",
        raw_data_path=raw_data_path,
        cleaned_data_path=None,
        model_metrics={},
        dataframe=df,  # Pass the loaded dataframe
        similar_past_tasks=[],
        run_id=run_id
    )
    
    memory_agent.start_timer()
    app = create_workflow()
    
    success = True
    error_msg = None
    
    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        success = False
        error_msg = str(e)
        print(f"❌ Pipeline failed: {error_msg}")
        raise
    
    memory_agent.after_pipeline(final_state, success, error_msg)
    
    return final_state
