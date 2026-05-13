from langgraph.graph import StateGraph, END
from anti_gravity.state import AntiGravityState
from anti_gravity.agents.quality_agent import data_quality_agent
from anti_gravity.agents.explorer_agent import explorer_agent
from anti_gravity.agents.trainer_agent import model_trainer_agent
from anti_gravity.agents.memory_agent import memory_agent

def create_workflow():
    """Create and compile the LangGraph workflow with memory"""
    
    workflow = StateGraph(AntiGravityState)
    
    # Add agent nodes
    workflow.add_node("memory_before", lambda state: memory_agent.before_pipeline(state))
    workflow.add_node("data_quality", data_quality_agent)
    workflow.add_node("explorer", explorer_agent)
    workflow.add_node("trainer", model_trainer_agent)
    workflow.add_node("memory_after", lambda state: {})  # Placeholder, actual save happens after
    
    # Define flow
    workflow.set_entry_point("memory_before")
    workflow.add_edge("memory_before", "data_quality")
    workflow.add_edge("data_quality", "explorer")
    workflow.add_edge("explorer", "trainer")
    workflow.add_edge("trainer", "memory_after")
    workflow.add_edge("memory_after", END)
    
    return workflow.compile()

def run_full_pipeline(raw_data_path: str, user_id: str = "default_user"):
    """Run the complete data science pipeline with memory"""
    
    initial_state = AntiGravityState(
        messages=[],
        dataset_info={},
        artifacts={},
        user_id=user_id,
        current_task="full_pipeline",
        raw_data_path=raw_data_path,
        cleaned_data_path=None,
        model_metrics={}
    )
    
    # Start timing
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
    
    # Store memories after pipeline
    memory_agent.after_pipeline(final_state, success, error_msg)
    
    return final_state
