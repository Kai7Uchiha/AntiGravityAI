import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from anti_gravity.state import AntiGravityState

def model_trainer_agent(state: AntiGravityState) -> dict:
    """Train a robust model using sklearn (no FLAML/LightGBM issues)"""
    print("🤖 Running Model Trainer Agent...")
    
    df = pd.read_csv(state["raw_data_path"])
    
    # Find target column (prefer columns with 'score' or last column)
    target_col = None
    for col in ['g6_consensus_score', 'target', 'label', 'score', 'consensus']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[-1]
    
    print(f"   🎯 Target column: {target_col}")
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Clean column names (no special chars)
    X.columns = [str(c).replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')[:50] for c in X.columns]
    
    # Handle all categorical columns
    print("   🔧 Encoding categorical variables...")
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        # Convert to string and clean values
        X[col] = X[col].astype(str).str.replace('[^a-zA-Z0-9]', '_', regex=True)
        # Label encode
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Handle numeric columns with missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Determine if classification or regression based on target values
    unique_targets = len(np.unique(y))
    unique_ratio = unique_targets / len(y)
    
    # Check if target is continuous (likely a score)
    is_continuous = False
    if y.dtype in ['float64', 'float32']:
        # Check if values have decimals (not just integers)
        if np.any(y != y.astype(int)):
            is_continuous = True
    elif unique_targets > 20:  # More than 20 unique values -> likely regression
        is_continuous = True
    
    # Check if it's actually categorical disguised as numbers
    if not is_continuous and unique_targets <= 10:
        # Convert to integer categories
        y = y.astype(int)
    
    if is_continuous:
        print(f"   📊 Regression task (continuous values: {y.min():.2f} to {y.max():.2f})")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        task = "regression"
        metric_name = "r2_score"
    else:
        print(f"   🎯 Classification task ({unique_targets} classes)")
        # Ensure y is integer for classification
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        task = "classification"
        metric_name = "accuracy"
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"   📊 Training on {X.shape[1]} features, {len(y)} samples")
    print("   🌲 Training Random Forest model...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    if task == "classification":
        score = accuracy_score(y_test, y_pred)
        print(f"\n   📋 Classification Report:")
        print(f"      Accuracy: {score:.3f}")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(5)
        
        print(f"   🔥 Top 5 important features:")
        for idx, row in feature_importance.iterrows():
            print(f"      - {row['feature'][:40]}: {row['importance']:.3f}")
    else:
        score = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"\n   📊 Regression Results:")
        print(f"      R² Score: {score:.3f}")
        print(f"      MSE: {mse:.3f}")
        print(f"      RMSE: {np.sqrt(mse):.3f}")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(5)
        
        print(f"   🔥 Top 5 important features:")
        for idx, row in feature_importance.iterrows():
            print(f"      - {row['feature'][:40]}: {row['importance']:.3f}")
    
    # Save model and preprocessing objects
    model_path = "model_rf.pkl"
    scaler_path = "scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Store artifacts
    if "artifacts" not in state:
        state["artifacts"] = {}
    state["artifacts"]["model"] = model_path
    state["artifacts"]["scaler"] = scaler_path
    state["model_metrics"] = {metric_name: score, "task": task}
    
    print(f"\n   ✅ Model saved: {model_path}")
    print(f"   🎯 {metric_name.capitalize()}: {score:.3f}")
    
    state["messages"].append({"role": "system", "content": f"Model trained with {metric_name} {score:.3f}"})
    
    return {"artifacts": state["artifacts"], "model_metrics": state["model_metrics"], "messages": state["messages"]}
