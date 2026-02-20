import os
import json
import requests

URL = "http://localhost:8000/predict"
ASSET_DIR = "tests/assets"

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]

results = []

# scan all images
for file_name in os.listdir(ASSET_DIR):

    file_path = os.path.join(ASSET_DIR, file_name)

    # skip non-images
    if not any(file_name.lower().endswith(ext) for ext in VALID_EXTENSIONS):
        continue

    # infer true label from filename
    if file_name.lower().startswith("cat"):
        true_label = "cat"
    elif file_name.lower().startswith("dog"):
        true_label = "dog"
    else:
        print(f"Skipping unknown label file: {file_name}")
        continue

    print(f"Testing: {file_name}")

    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(URL, files=files)

        pred = response.json()["prediction"]

        results.append({
            "image": file_name,
            "true_label": true_label,
            "predicted_label": pred
        })

    except Exception as e:
        print(f"Failed on {file_name}: {e}")

# save results
with open("deployment_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} predictions to deployment_results.json")
