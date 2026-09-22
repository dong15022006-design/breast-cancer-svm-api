import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True

def test_metadata_has_30_features():
    r = client.get("/metadata")
    assert r.status_code == 200
    assert len(r.json()["feature_names"]) == 30

def test_predict_rejects_missing_features():
    r = client.post("/predict", json={"features": {"mean radius": 10.0}})
    assert r.status_code == 422

def test_predict_happy_path():
    from sklearn.datasets import load_breast_cancer
    bundle = load_breast_cancer(as_frame=True)
    sample = bundle.data.iloc[0].to_dict()
    payload = {"features": {k: float(v) for k, v in sample.items()}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_label"] in ("malignant", "benign")
    assert 0.0 <= body["probability_malignant"] <= 1.0
    assert 0.0 <= body["probability_benign"] <= 1.0
