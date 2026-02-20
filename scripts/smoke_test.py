import time
import requests

def wait_for_service():
    for _ in range(10):
        try:
            r = requests.get("http://localhost:8000/health")
            if r.status_code == 200:
                return True
        except:
            time.sleep(3)
    return False

def run_smoke_test():
    url = "http://localhost:8000/predict"

    files = {
        "file": open("sample.jpg", "rb")
    }

    response = requests.post(url, files=files)

    assert response.status_code == 200


if __name__ == "__main__":
    wait_for_service()
    run_smoke_test()
