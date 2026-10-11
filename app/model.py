from functools import lru_cache

from transformers import pipeline

MODEL_NAME = "s-nlp/russian_toxicity_classifier"


# Загрузка и кэширование модели + пайплайн
@lru_cache(maxsize=1)
def get_classifier():
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        truncation=True,
    )


# Возвращает класс и уверенность модели для одного текста
def predict_text(text: str) -> dict:
    result = get_classifier()(text)[0]

    return {
        "label": str(result["label"]),
        "probability": float(result["score"]),
    }


# Классифицирует список текстов
def predict_batch(texts: list[str]) -> list[dict]:
    if not texts:
        return []

    results = get_classifier()(
        texts,
        batch_size=8,
    )

    return [
        {
            "label": str(result["label"]),
            "probability": float(result["score"]),
        }
        for result in results
    ]
