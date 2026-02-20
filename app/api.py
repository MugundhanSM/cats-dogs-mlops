import time
import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException

from src.inference import Predictor
from app.schemas import PredictionResponse
from app.logger import get_logger

app = FastAPI(title="Cats vs Dogs Classifier")

logger = get_logger()
predictor = Predictor()

request_count = 0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    global request_count
    request_count += 1
    logger.info(f"Total Requests: {request_count}")

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid image format")

    temp_path = f"temp_{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predictor.predict(temp_path)

        latency = (time.time() - start_time) * 1000

        logger.info(
            f"Prediction: {result['label']} | Confidence: {result['confidence']:.3f} | Latency: {latency:.2f} ms"
        )

        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            latency_ms=latency
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
