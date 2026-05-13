import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from anti_gravity.state import AntiGravityState
from anti_gravity.tools.data_tools import generate_profile, quick_plot

def save_plot_safe(df: pd.DataFrame, col: str) -> str:
    """Wrapper for quick_plot with safe filename"""
    safe_name = str(col).replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    plot_path = f"plot_{safe_name}.png"
    quick_plot(df, col, save_path=plot_path)
    return plot_path

def explorer_agent(state: AntiGravityState) -> dict:
    """Perform EDA with parallel plotting"""
    print("📊 Running Explorer/EDA Agent...")
    
    # Use cleaned dataframe from state
    df = state["dataframe"]
    
    print(f"   📋 Dataset Overview:")
    print(f"      - Shape: {df.shape}")
    print(f"      - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Generate text profile report
    profile_path = generate_profile(df, "EDA_Report")
    
    # Parallel plotting for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plot_paths = []
    
    if len(numeric_cols) > 0:
        cols_to_plot = list(numeric_cols[:5])  # First 5 numeric columns
        print(f"   🎨 Generating {len(cols_to_plot)} plots in parallel...")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(save_plot_safe, df, col): col for col in cols_to_plot}
            for future in as_completed(futures):
                plot_path = future.result()
                plot_paths.append(plot_path)
                print(f"      ✅ Generated: {plot_path}")
    
    # Store artifacts
    if "artifacts" not in state:
        state["artifacts"] = {}
    state["artifacts"]["profile_report"] = profile_path
    state["artifacts"]["plots"] = plot_paths
    
    state["messages"].append({"role": "system", "content": f"EDA completed with {len(plot_paths)} plots"})
    
    return {"artifacts": state["artifacts"], "messages": state["messages"]}
