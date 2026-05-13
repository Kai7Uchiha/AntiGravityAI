"""
Test supervised workflow with routing and critic
"""

from anti_gravity.workflows.main_workflow import run_full_pipeline
from sklearn.datasets import load_iris
import pandas as pd

# Prepare dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.to_csv("iris.csv", index=False)

print("="*60)
print("🤖 Testing Supervised Workflow")
print("="*60)

result = run_full_pipeline("iris.csv", user_id="test_user")

print("\n" + "="*60)
print("✅ Supervised workflow complete!")
print(f"Model accuracy: {result['model_metrics'].get('accuracy', 'N/A')}")
print(f"Total messages: {len(result['messages'])}")
print("\nMessage timeline:")
for msg in result['messages'][-5:]:  # Last 5 messages
    print(f"  - {msg.get('role', 'unknown')}: {msg.get('content', '')[:80]}")
print("="*60)
