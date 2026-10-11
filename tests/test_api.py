from fastapi.testclient import TestClient

from app.main import app

# Экземпляр тестового клиента
client = TestClient(app)


def test_health_returns_ok_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_prediction_schema():
    response = client.post(
        "/predict",
        json={"text": "Сегодня прекрасная погода!"},
    )

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["label"], str)
    assert isinstance(data["probability"], (int, float))
    assert 0 <= data["probability"] <= 1


def test_predict_rejects_missing_text():
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_predict_batch_returns_prediction_for_each_text():
    texts = [
        "Хорошего тебе дня!",
        "Ты замечательный человек.",
        "Это ужасное сообщение.",
    ]

    response = client.post(
        "/predict_batch",
        json={"texts": texts},
    )

    assert response.status_code == 200

    data = response.json()
    assert len(data["predictions"]) == len(texts)

    for prediction in data["predictions"]:
        assert isinstance(prediction["label"], str)
        assert isinstance(prediction["probability"], (int, float))
        assert 0 <= prediction["probability"] <= 1


def test_predict_batch_accepts_empty_list():
    response = client.post(
        "/predict_batch",
        json={"texts": []},
    )

    assert response.status_code == 200
    assert response.json() == {"predictions": []}


# python -m pytest -v
