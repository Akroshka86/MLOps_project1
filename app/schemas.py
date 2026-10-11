from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


# Структура запроса и ответа
class PredictRequest(BaseModel):
    text: str = Field(min_length=1, description="Текст для классификации")


class PredictionResponse(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(description="Список текстов для классификации")


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    name: str
    task: str
    transformers_version: str
    loaded: bool
