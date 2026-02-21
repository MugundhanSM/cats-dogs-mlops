import os
import json
import requests

URL = "http://localhost:8000/predict"
ASSET_DIR = "data/val"

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]

results = []

print("Current working dir:", os.getcwd())
print("Scanning:", ASSET_DIR)

for root, _, files in os.walk(ASSET_DIR):

    for file_name in files:

        if not any(file_name.lower().endswith(ext) for ext in VALID_EXTENSIONS):
            continue

        file_path = os.path.join(root, file_name)

        # infer label from folder name OR filename
        folder_name = os.path.basename(root).lower()

        if "cat" in folder_name or file_name.lower().startswith("cat"):
            true_label = "cat"
        elif "dog" in folder_name or file_name.lower().startswith("dog"):
            true_label = "dog"
        else:
            print(f"Skipping unknown label: {file_path}")
            continue

        print(f"Testing: {file_path}")

        try:
            with open(file_path, "rb") as f:
                response = requests.post(URL, files={"file": f})

            pred = response.json().get("prediction")

            results.append({
                "image": file_path,
                "true_label": true_label,
                "predicted_label": pred
            })

        except Exception as e:
            print(f"Failed on {file_path}: {e}")

# save results
with open("deployment_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} predictions to deployment_results.json")
