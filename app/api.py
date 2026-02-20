import time
import uuid
import logging
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
# Monitoring counters
# -------------------------
request_count = 0
total_latency = 0.0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    avg_latency = (
        total_latency / request_count if request_count > 0 else 0
    )

    return {
        "total_requests": request_count,
        "average_latency_ms": round(avg_latency, 2),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global request_count, total_latency

    start_time = time.time()
    request_id = str(uuid.uuid4())

    # save temp file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # model prediction
    result = predictor.predict(temp_path)

    latency_ms = (time.time() - start_time) * 1000

    # update metrics
    request_count += 1
    total_latency += latency_ms

    # logging (NO sensitive data)
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
