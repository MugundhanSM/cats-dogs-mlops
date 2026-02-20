import os
import re
import json
import time
import requests

URL = "http://localhost:8000/predict"
ASSET_DIR = "tests/assets"

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]

results = []

print("Current working dir:", os.getcwd())
print("Asset path:", ASSET_DIR)

if not os.path.exists(ASSET_DIR):
    print("ASSET DIR NOT FOUND")
    exit(1)

print("Files found:", os.listdir(ASSET_DIR))


def extract_label(filename):
    """
    Extract true label from filename.
    Works for:
        cat.4001.jpg
        dog_12.png
        cat-99.jpeg
    """
    match = re.match(r"(cat|dog)", filename.lower())
    return match.group(1) if match else None


# scan all images
for file_name in os.listdir(ASSET_DIR):

    file_path = os.path.join(ASSET_DIR, file_name)

    # skip non-images
    if not any(file_name.lower().endswith(ext) for ext in VALID_EXTENSIONS):
        continue

    # infer true label
    true_label = extract_label(file_name)
    if not true_label:
        print(f"Skipping unknown label file: {file_name}")
        continue

    print(f"Testing: {file_name}")

    try:
        start = time.time()

        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(URL, files=files, timeout=20)

        latency_ms = round((time.time() - start) * 1000, 2)

        if response.status_code != 200:
            print(f"API failed for {file_name}: {response.status_code}")
            continue

        data = response.json()

        pred = data.get("prediction") or data.get("label")
        confidence = data.get("confidence", None)

        results.append({
            "image": file_name,
            "true_label": true_label,
            "predicted_label": pred,
            "confidence": confidence,
            "latency_ms": latency_ms
        })

    except Exception as e:
        print(f"Failed on {file_name}: {e}")


# save results
with open("deployment_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} predictions to deployment_results.json")
