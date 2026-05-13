#!/usr/bin/env python
"""
AntiGravity AI - Main entry point with Kaggle dataset support
Run: python main.py
"""

import pandas as pd
from anti_gravity.workflows.main_workflow import run_full_pipeline

# Try to import Kaggle loader (optional)
try:
    from anti_gravity.tools.kaggle_loader import get_personality_emotion_dataset
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("⚠️  kagglehub not installed. Install with: pip install kagglehub")

def prepare_iris_dataset():
    """Download and save Iris dataset"""
    print("📦 Preparing Iris dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    df = pd.read_csv(url, header=None, names=columns)
    df.to_csv("iris.csv", index=False)
    print("   ✅ Iris dataset saved to iris.csv")
    return "iris.csv"

def prepare_titanic_dataset():
    """Download Titanic dataset from URL"""
    print("📦 Preparing Titanic dataset...")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    # Keep only numeric columns for simplicity
    df = df.select_dtypes(include=['int64', 'float64'])
    df.to_csv("titanic.csv", index=False)
    print("   ✅ Titanic dataset saved to titanic.csv")
    return "titanic.csv"

def prepare_kaggle_personality_dataset():
    """Download and prepare the Personality-Emotion dataset from Kaggle"""
    print("📦 Preparing Kaggle Personality-Emotion dataset...")
    try:
        df = get_personality_emotion_dataset()
        
        # Save to CSV
        output_path = "personality_emotion.csv"
        df.to_csv(output_path, index=False)
        print(f"   ✅ Dataset saved to {output_path}")
        
        # Print dataset info
        print(f"\n   📊 Dataset Overview:")
        print(f"      - Shape: {df.shape}")
        print(f"      - Columns: {list(df.columns)[:5]}...")  # First 5 columns
        print(f"      - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return output_path
    except Exception as e:
        print(f"   ❌ Error loading Kaggle dataset: {e}")
        print("   Falling back to Iris dataset...")
        return prepare_iris_dataset()

def main():
    print("="*60)
    print("🚀 AntiGravity AI - Autonomous Data Science System")
    print("="*60)
    
    # Dataset selection
    print("\nChoose a dataset:")
    print("1. Iris (simple, fast, built-in)")
    print("2. Titanic (real-world, URL)")
    if KAGGLE_AVAILABLE:
        print("3. Kaggle: Personality-Emotion Dataset (recommended for Phase 1 testing)")
    
    choice = input("\nEnter 1, 2, or 3: ").strip()
    
    if choice == "1":
        data_path = prepare_iris_dataset()
    elif choice == "2":
        data_path = prepare_titanic_dataset()
    elif choice == "3" and KAGGLE_AVAILABLE:
        data_path = prepare_kaggle_personality_dataset()
    else:
        print("Invalid choice. Using Iris dataset.")
        data_path = prepare_iris_dataset()
    
    print("\n" + "="*60)
    print("Starting autonomous data science pipeline...")
    print("="*60 + "\n")
    
    # Run the pipeline
    final_state = run_full_pipeline(data_path, user_id="test_user")
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📊 Final Results:")
    print(f"   Dataset: {final_state['dataset_info']['shape'][0]} rows, {final_state['dataset_info']['shape'][1]} columns")
    
    # Handle both classification and regression metrics
    if 'accuracy' in final_state['model_metrics']:
        print(f"   Model Accuracy: {final_state['model_metrics']['accuracy']:.3f}")
    elif 'r2_score' in final_state['model_metrics']:
        print(f"   Model R² Score: {final_state['model_metrics']['r2_score']:.3f}")
    elif 'score' in final_state['model_metrics']:
        print(f"   Model Score: {final_state['model_metrics']['score']:.3f}")
    
    print(f"\n📁 Generated Files:")
    for name, path in final_state['artifacts'].items():
        print(f"   - {name}: {path}")
    
    # Show sample of dataset
    df = pd.read_csv(data_path)
    print(f"\n📋 Dataset Preview (first 5 rows):")
    print(df.head())
    
    print(f"\n📊 Column Info (first 5 columns):")
    for col in df.columns[:5]:
        print(f"   - {col}: {df[col].dtype}")

if __name__ == "__main__":
    main()
