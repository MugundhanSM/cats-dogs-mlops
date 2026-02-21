import time
import uuid
import logging
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File
from src.inference import Predictor

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_api")

app = FastAPI()
predictor = Predictor()

# -------------------------
# Monitoring metrics
# -------------------------
metrics_data = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_latency": 0.0,
    "max_latency": 0.0,
    "min_latency": float("inf"),
    "last_request_time": None,
    "prediction_counts": defaultdict(int),
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():

    total = metrics_data["total_requests"]

    avg_latency = (
        metrics_data["total_latency"] / total if total > 0 else 0
    )

    min_latency = (
        metrics_data["min_latency"]
        if metrics_data["min_latency"] != float("inf")
        else 0
    )

    return {
        "total_requests": total,
        "successful_requests": metrics_data["successful_requests"],
        "failed_requests": metrics_data["failed_requests"],
        "average_latency_ms": round(avg_latency, 2),
        "max_latency_ms": round(metrics_data["max_latency"], 2),
        "min_latency_ms": round(min_latency, 2),
        "prediction_counts": dict(metrics_data["prediction_counts"]),
        "last_request_time": metrics_data["last_request_time"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    metrics_data["total_requests"] += 1
    metrics_data["last_request_time"] = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    try:
        # save temp file
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # prediction
        result = predictor.predict(temp_path)

        latency_ms = (time.time() - start_time) * 1000

        # update latency metrics
        metrics_data["total_latency"] += latency_ms
        metrics_data["max_latency"] = max(
            metrics_data["max_latency"], latency_ms
        )
        metrics_data["min_latency"] = min(
            metrics_data["min_latency"], latency_ms
        )

        metrics_data["successful_requests"] += 1
        metrics_data["prediction_counts"][result["label"]] += 1

        logger.info(
            f"request_id={request_id} "
            f"label={result['label']} "
            f"confidence={result['confidence']:.4f} "
            f"latency_ms={latency_ms:.2f}"
        )

        return {
            "request_id": request_id,
            "prediction": result["label"],
            "confidence": result["confidence"],
            "latency_ms": round(latency_ms, 2),
        }

    except Exception as e:
        metrics_data["failed_requests"] += 1

        logger.error(
            f"request_id={request_id} error={str(e)}"
        )

        return {
            "request_id": request_id,
            "error": "prediction failed",
        }
