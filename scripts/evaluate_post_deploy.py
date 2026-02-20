import json

with open("deployment_results.json") as f:
    data = json.load(f)

correct = sum(
    1 for r in data if r["true_label"] == r["predicted_label"]
)

accuracy = correct / len(data)

print("Post-deployment accuracy:", accuracy)
