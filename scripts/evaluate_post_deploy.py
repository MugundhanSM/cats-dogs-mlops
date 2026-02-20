import json
from collections import defaultdict

INPUT_FILE = "deployment_results.json"


def safe_get(record, key):
    return record.get(key, None)


with open(INPUT_FILE) as f:
    data = json.load(f)

if not data:
    print("No deployment results found.")
    exit(0)

total = 0
correct = 0

per_class_total = defaultdict(int)
per_class_correct = defaultdict(int)

latencies = []

for r in data:
    true_label = safe_get(r, "true_label")
    pred_label = safe_get(r, "predicted_label")
    latency = safe_get(r, "latency_ms")

    if true_label is None or pred_label is None:
        continue

    total += 1
    per_class_total[true_label] += 1

    if true_label == pred_label:
        correct += 1
        per_class_correct[true_label] += 1

    if latency is not None:
        latencies.append(latency)


# ---- metrics ----
accuracy = correct / total if total > 0 else 0

print("\n===== POST DEPLOYMENT EVALUATION =====")
print(f"Total samples: {total}")
print(f"Correct predictions: {correct}")
print(f"Overall accuracy: {accuracy:.4f}")

print("\nPer-class accuracy:")
for cls in sorted(per_class_total.keys()):
    cls_acc = (
        per_class_correct[cls] / per_class_total[cls]
        if per_class_total[cls] > 0 else 0
    )
    print(f"  {cls}: {cls_acc:.4f} ({per_class_correct[cls]}/{per_class_total[cls]})")

if latencies:
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage latency: {avg_latency:.2f} ms")

print("======================================")
