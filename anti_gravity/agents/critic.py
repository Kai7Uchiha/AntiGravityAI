"""
Critic Agent - Reviews outputs and suggests improvements
Enables self-reflection and iterative refinement
"""
from typing import Dict

class CriticAgent:
    """Reviews agent outputs and provides feedback"""
    
    def __init__(self):
        self.max_iterations = 3
        self.iteration_count = 0
    
    def review_quality_agent(self, state: Dict) -> Dict:
        """Review data quality report and suggest fixes"""
        
        dataset_info = state.get("dataset_info", {})
        missing_values = dataset_info.get("missing", {})
        
        suggestions = []
        
        # Check for high missing values
        high_missing = {k: v for k, v in missing_values.items() if v > 0}
        if high_missing:
            suggestions.append(f"Found missing values in: {list(high_missing.keys())}")
            suggestions.append("Recommendation: Use auto_clean() to fill missing values")
        
        # Check for duplicates
        if dataset_info.get("duplicates", 0) > 0:
            suggestions.append(f"Found {dataset_info['duplicates']} duplicate rows")
            suggestions.append("Recommendation: Remove duplicates with drop_duplicates()")
        
        # Check for data types
        dtypes = dataset_info.get("dtypes", {})
        object_cols = [k for k, v in dtypes.items() if v == 'object']
        if len(object_cols) > 3:
            suggestions.append(f"Found {len(object_cols)} categorical columns")
            suggestions.append("Recommendation: Consider encoding for ML models")
        
        review = {
            "score": 0.8 if not suggestions else 0.5,
            "suggestions": suggestions,
            "passed": len(suggestions) == 0
        }
        
        if suggestions:
            print("   🔍 Critic feedback:")
            for s in suggestions:
                print(f"      - {s}")
        
        return review
    
    def review_model_agent(self, state: Dict) -> Dict:
        """Review model performance and suggest improvements"""
        
        metrics = state.get("model_metrics", {})
        accuracy = metrics.get("accuracy", metrics.get("r2_score", 0))
        task = metrics.get("task", "unknown")
        
        suggestions = []
        
        # Check performance
        if task == "classification":
            if accuracy < 0.7:
                suggestions.append(f"Low accuracy: {accuracy:.3f}")
                suggestions.append("Recommendation: Try feature engineering or more data")
            elif accuracy > 0.95:
                suggestions.append(f"Excellent accuracy: {accuracy:.3f}")
                suggestions.append("Recommendation: Consider checking for overfitting")
        else:  # regression
            if accuracy < 0.5:
                suggestions.append(f"Low R² score: {accuracy:.3f}")
                suggestions.append("Recommendation: Add more features or try different model")
        
        # Check if model was saved
        if "model" not in state.get("artifacts", {}):
            suggestions.append("Model not saved!")
            suggestions.append("Recommendation: Save model for future use")
        
        review = {
            "score": accuracy,
            "suggestions": suggestions,
            "passed": len(suggestions) == 0,
            "needs_retraining": accuracy < 0.5
        }
        
        if suggestions:
            print("   🔍 Critic feedback:")
            for s in suggestions:
                print(f"      - {s}")
        
        return review

# Global instance
critic = CriticAgent()
