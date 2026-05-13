"""
Parallel Model Trainer - Trains multiple models concurrently
Uses multiprocessing for true parallelism
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib

class ParallelTrainer:
    """Trains multiple models in parallel and selects the best"""
    
    def __init__(self, n_jobs=4):
        self.n_jobs = n_jobs
        self.models = {}
    
    def get_models_for_task(self, task_type: str, X_shape: tuple):
        """Get appropriate models based on task type"""
        
        n_samples, n_features = X_shape
        
        if task_type == "classification":
            return {
                "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "SVM": SVC(random_state=42)
            }
        else:  # regression
            return {
                "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
                "Linear Regression": LinearRegression(),
                "SVR": SVR()
            }
    
    def train_single_model(self, name: str, model, X_train, y_train, X_test, y_test, task_type: str):
        """Train a single model and return performance"""
        try:
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            if task_type == "classification":
                score = model.score(X_test, y_test)
                metric = "accuracy"
            else:
                score = model.score(X_test, y_test)
                metric = "r2"
            
            return {
                "name": name,
                "model": model,
                "score": score,
                "metric": metric,
                "success": True
            }
        except Exception as e:
            return {
                "name": name,
                "error": str(e),
                "success": False
            }
    
    def train_all_parallel(self, X_train, y_train, X_test, y_test, task_type: str):
        """Train all models in parallel"""
        
        models = self.get_models_for_task(task_type, X_train.shape)
        results = []
        
        print(f"   🚀 Training {len(models)} models in parallel...")
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(
                    self.train_single_model, 
                    name, model, X_train, y_train, X_test, y_test, task_type
                ): name 
                for name, model in models.items()
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result["success"]:
                    print(f"      ✅ {result['name']}: {result['metric']}={result['score']:.3f}")
                else:
                    print(f"      ❌ {result['name']}: {result.get('error', 'Unknown error')}")
        
        # Select best model
        successful = [r for r in results if r["success"]]
        if successful:
            best = max(successful, key=lambda x: x["score"])
            return best, results
        
        return None, results

# Global instance
parallel_trainer = ParallelTrainer(n_jobs=2)  # Use 2 threads for Codespace
