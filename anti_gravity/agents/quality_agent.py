import pandas as pd
import numpy as np
from anti_gravity.state import AntiGravityState

def data_quality_agent(state: AntiGravityState) -> dict:
    """Analyze data quality: shape, missing values, duplicates, types"""
    print("🔍 Running Data Quality Agent...")
    
    df = pd.read_csv(state["raw_data_path"])
    
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
    
    # Analyze target column (last column or score column)
    target_col = None
    for col in ['g6_consensus_score', 'target', 'label', 'score', 'consensus']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[-1]
    
    print(f"\n   📊 Target column analysis ('{target_col}'):")
    print(f"      Type: {df[target_col].dtype}")
    print(f"      Unique values: {df[target_col].nunique()}")
    
    # Check if continuous or discrete
    if df[target_col].dtype in ['float64', 'float32']:
        print(f"      Range: {df[target_col].min():.2f} to {df[target_col].max():.2f}")
        print(f"      Mean: {df[target_col].mean():.2f}")
        print(f"      Std: {df[target_col].std():.2f}")
    else:
        print(f"      Value counts: {df[target_col].value_counts().head(3).to_dict()}")
    
    state["dataset_info"] = report
    state["messages"].append({"role": "system", "content": f"Data quality check: {report}"})
    
    return {"dataset_info": report, "messages": state["messages"]}
