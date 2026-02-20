from pydantic import BaseModel

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float
