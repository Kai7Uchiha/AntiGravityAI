import pandas as pd
from anti_gravity.state import AntiGravityState
from anti_gravity.tools.data_tools import generate_profile, quick_plot

def explorer_agent(state: AntiGravityState) -> dict:
    """Perform EDA: profiling, visualizations"""
    print("📊 Running Explorer/EDA Agent...")
    
    df = pd.read_csv(state["raw_data_path"])
    
    # Show dataset overview
    print(f"   📋 Dataset Overview:")
    print(f"      - Shape: {df.shape}")
    print(f"      - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"      - Columns: {', '.join(df.columns[:5])}...")
    
    # Generate text profile report
    profile_path = generate_profile(df, "EDA_Report")
    
    # Create plots for first few numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        for i, col in enumerate(list(numeric_cols)[:3]):  # First 3 numeric columns
            # Clean filename
            safe_name = col.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
            plot_path = f"plot_{safe_name}.png"
            quick_plot(df, col, save_path=plot_path)
    
    # Store artifacts
    if "artifacts" not in state:
        state["artifacts"] = {}
    state["artifacts"]["profile_report"] = profile_path
    
    state["messages"].append({"role": "system", "content": f"EDA completed. Report: {profile_path}"})
    
    return {"artifacts": state["artifacts"], "messages": state["messages"]}
