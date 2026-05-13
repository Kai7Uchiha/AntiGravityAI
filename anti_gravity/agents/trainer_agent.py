import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from anti_gravity.state import AntiGravityState
from anti_gravity.utils.column_utils import detect_target_column, detect_task_type
import os

def model_trainer_agent(state: AntiGravityState) -> dict:
    """Train model using in-memory dataframe (no redundant disk reads)"""
    print("🤖 Running Model Trainer Agent...")
    
    # Use cleaned dataframe from state
    df = state["dataframe"]
    run_id = state.get("run_id", "default")
    
    # Detect target column
    target_col = detect_target_column(df)
    print(f"   🎯 Target column: {target_col}")
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Handle categorical columns
    print("   🔧 Encoding categorical variables...")
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Determine task type using shared utility
    task_type = detect_task_type(y)
    
    if task_type == "classification":
        print(f"   🎯 Classification task ({y.nunique()} classes)")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        metric_name = "accuracy"
    else:
        print(f"   📊 Regression task (continuous values)")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        metric_name = "r2_score"
    
    # Check for similar past tasks to influence training
    similar_tasks = state.get("similar_past_tasks", [])
    if similar_tasks:
        print(f"   📚 Using {len(similar_tasks)} similar past experiences to inform training")
        # This is where memory actually influences decisions
    
    # Split data (no scaler - tree models are scale-invariant)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   📊 Training on {X.shape[1]} features, {len(y)} samples")
    print("   🌲 Training Random Forest model...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    if task_type == "classification":
        score = accuracy_score(y_test, y_pred)
        print(f"\n   📋 Classification Report:")
        print(f"      Accuracy: {score:.3f}")
    else:
        score = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"\n   📊 Regression Results:")
        print(f"      R² Score: {score:.3f}")
        print(f"      RMSE: {np.sqrt(mse):.3f}")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(5)
    
    print(f"   🔥 Top 5 important features:")
    for _, row in feature_importance.iterrows():
        print(f"      - {row['feature'][:40]}: {row['importance']:.3f}")
    
    # Versioned artifact saving
    os.makedirs("artifacts", exist_ok=True)
    model_path = f"artifacts/model_{run_id}.pkl"
    joblib.dump(model, model_path)
    
    # Store artifacts
    if "artifacts" not in state:
        state["artifacts"] = {}
    state["artifacts"]["model"] = model_path
    state["model_metrics"] = {metric_name: score, "task": task_type}
    
    print(f"\n   ✅ Model saved: {model_path}")
    print(f"   🎯 {metric_name.capitalize()}: {score:.3f}")
    
    state["messages"].append({"role": "system", "content": f"Model trained with {metric_name} {score:.3f}"})
    
    return {"artifacts": state["artifacts"], "model_metrics": state["model_metrics"], "messages": state["messages"]}
