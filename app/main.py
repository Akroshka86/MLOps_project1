import transformers
from fastapi import FastAPI

from app.model import MODEL_NAME, get_classifier, predict_batch, predict_text
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    PredictRequest,
)

app = FastAPI(
    title="Russian Toxicity Classification API",
    description="REST API для классификации русскоязычных текстов.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    return predict_text(request.text)


@app.post(
    "/predict_batch",
    response_model=BatchPredictResponse,
)
def predict_batch_endpoint(request: BatchPredictRequest):
    predictions = predict_batch(request.texts)
    return {"predictions": predictions}


@app.get("/model_info", response_model=ModelInfoResponse)
def model_info():
    return {
        "name": MODEL_NAME,
        "task": "Russian text toxicity classification",
        "transformers_version": transformers.__version__,
        "loaded": get_classifier.cache_info().currsize > 0,
    }
