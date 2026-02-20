import time

def wait_for_service():
    for _ in range(10):
        try:
            r = requests.get("http://localhost:8000/health")
            if r.status_code == 200:
                return True
        except:
            time.sleep(3)
    return False


def test_prediction():
    with open("tests/sample.jpg", "rb") as f:
        files = {"file": f}
        r = requests.post("http://localhost:8000/predict", files=files)
        assert r.status_code == 200