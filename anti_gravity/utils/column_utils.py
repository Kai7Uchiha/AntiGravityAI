"""Shared utilities for column detection and task type"""

import pandas as pd
import numpy as np

TARGET_PRIORITY = ['g6_consensus_score', 'target', 'label', 'score', 'consensus']

def detect_target_column(df: pd.DataFrame) -> str:
    """Detect target column based on priority list"""
    for col in TARGET_PRIORITY:
        if col in df.columns:
            return col
    return df.columns[-1]  # fallback to last column

def detect_task_type(series: pd.Series) -> str:
    """Returns 'regression' or 'classification' based on data"""
    if series.dtype in ['float64', 'float32']:
        # Check if values have decimals (not just integers)
        if (series != series.astype(int)).any():
            return 'regression'
    
    # More than 20 unique values -> likely regression
    if series.nunique() > 20:
        return 'regression'
    
    return 'classification'

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names for ML compatibility"""
    df = df.copy()
    df.columns = [
        str(c).replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')[:50]
        for c in df.columns
    ]
    return df
