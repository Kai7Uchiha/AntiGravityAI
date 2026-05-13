"""
Test script for memory system
Run: python test_memory.py
"""

from anti_gravity.workflows.main_workflow import run_full_pipeline

def test_memory():
    print("="*60)
    print("🧪 Testing Memory System")
    print("="*60)
    
    # First run - should have no memories
    print("\n--- RUN 1: First execution (no prior memory) ---")
    result1 = run_full_pipeline("iris.csv", user_id="test_user_1")
    print(f"Run 1 R² Score: {result1['model_metrics'].get('r2_score', result1['model_metrics'].get('accuracy', 'N/A'))}")
    
    # Second run - should find memories from first
    print("\n--- RUN 2: Second execution (with memory retrieval) ---")
    result2 = run_full_pipeline("iris.csv", user_id="test_user_1")
    print(f"Run 2 R² Score: {result2['model_metrics'].get('r2_score', result2['model_metrics'].get('accuracy', 'N/A'))}")
    
    # Different user - should NOT find memories
    print("\n--- RUN 3: Different user (isolated memory) ---")
    result3 = run_full_pipeline("iris.csv", user_id="test_user_2")
    print(f"Run 3 R² Score: {result3['model_metrics'].get('r2_score', result3['model_metrics'].get('accuracy', 'N/A'))}")
    
    print("\n" + "="*60)
    print("✅ Memory test complete!")
    print("="*60)

if __name__ == "__main__":
    # First make sure iris dataset exists
    from sklearn.datasets import load_iris
    import pandas as pd
    
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv("iris.csv", index=False)
    
    test_memory()
