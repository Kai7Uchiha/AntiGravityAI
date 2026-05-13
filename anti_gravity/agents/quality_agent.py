import pandas as pd
import numpy as np
from anti_gravity.state import AntiGravityState
from anti_gravity.utils.column_utils import detect_target_column, detect_task_type, clean_column_names
from anti_gravity.tools.data_tools import auto_clean

def data_quality_agent(state: AntiGravityState) -> dict:
    """Analyze data quality and save cleaned version"""
    print("🔍 Running Data Quality Agent...")
    
    # Use dataframe from state (already loaded)
    df = state["dataframe"].copy()
    
    # Clean column names
    df = clean_column_names(df)
    
    # Basic info
    report = {
        "shape": df.shape,
        "missing": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    
    print(f"   Dataset shape: {report['shape']}")
    print(f"   Duplicate rows: {report['duplicates']}")
    
    missing_cols = {k: v for k, v in report['missing'].items() if v > 0}
    if missing_cols:
        print(f"   Columns with missing values: {missing_cols}")
    else:
        print("   ✅ No missing values found")
    
    # Analyze target column
    target_col = detect_target_column(df)
    task_type = detect_task_type(df[target_col])
    
    print(f"\n   📊 Target column analysis ('{target_col}'):")
    print(f"      Type: {df[target_col].dtype}")
    print(f"      Task: {task_type}")
    print(f"      Unique values: {df[target_col].nunique()}")
    
    if df[target_col].dtype in ['float64', 'float32']:
        print(f"      Range: {df[target_col].min():.2f} to {df[target_col].max():.2f}")
        print(f"      Mean: {df[target_col].mean():.2f}")
    
    # Auto-clean and save once
    print("\n   🧹 Running auto_clean...")
    cleaned_df = auto_clean(df)
    cleaned_path = state["raw_data_path"].replace(".csv", "_cleaned.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"   ✅ Cleaned dataset saved to: {cleaned_path}")
    
    state["dataset_info"] = report
    state["cleaned_data_path"] = cleaned_path
    state["dataframe"] = cleaned_df  # Update state with cleaned version
    state["messages"].append({"role": "system", "content": f"Data quality check complete. Task type: {task_type}"})
    
    return {
        "dataset_info": report,
        "cleaned_data_path": cleaned_path,
        "dataframe": cleaned_df,
        "messages": state["messages"]
    }
