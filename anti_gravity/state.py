from typing import TypedDict, List, Dict, Any, Optional

class AntiGravityState(TypedDict):
    """Shared state that flows through all agents"""
    messages: List[Dict[str, str]]
    dataset_info: Dict[str, Any]
    artifacts: Dict[str, str]
    user_id: str
    current_task: str
    raw_data_path: Optional[str]
    cleaned_data_path: Optional[str]
    model_metrics: Dict[str, float]
    dataframe: Optional[Any]  # Holds pd.DataFrame in memory during pipeline
    similar_past_tasks: Optional[List[Dict]]  # Retrieved memories
    run_id: Optional[str]  # Unique ID for this run
