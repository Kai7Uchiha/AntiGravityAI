#!/usr/bin/env python
"""
Run tests: pytest test_workflow.py -v
"""

import pytest
import pandas as pd
import os
from anti_gravity.tools.data_tools import auto_clean, generate_profile
from anti_gravity.workflows.main_workflow import run_full_pipeline

def test_auto_clean():
    """Test that auto_clean removes all missing values"""
    df_test = pd.DataFrame({'A': [1, None, 3], 'B': ['x', None, 'z']})
    cleaned = auto_clean(df_test)
    assert cleaned.isnull().sum().sum() == 0
    print("✓ Auto-clean test passed")

def test_profile_generation():
    """Test that profile report is created"""
    df = pd.DataFrame({'col1': [1,2,3], 'col2': ['a','b','c']})
    filename = generate_profile(df, "test")
    assert os.path.exists(filename)
    os.remove(filename)
    print("✓ Profile generation test passed")

def test_full_pipeline_iris():
    """Test full pipeline on Iris dataset"""
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv("test_iris.csv", index=False)
    
    final_state = run_full_pipeline("test_iris.csv", user_id="test")
    
    assert "accuracy" in final_state["model_metrics"]
    assert final_state["model_metrics"]["accuracy"] > 0.9
    assert os.path.exists("model.pkl")
    
    # Cleanup
    os.remove("test_iris.csv")
    print("✓ Full pipeline test passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])