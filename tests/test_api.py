import pytest
from fastapi.testclient import TestClient

from creditrisk import api
from creditrisk.config import FEATURES
from creditrisk.data import load_data


@pytest.fixture(scope="module")
def client(artifacts_dir, monkeypatch_module):
    monkeypatch_module.setenv("CREDITRISK_ARTIFACTS", str(artifacts_dir))
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def _company(label: int):
    df = load_data()
    row = df[df["bankrupt"] == label].iloc[0]
    return {f: (None if row[f] != row[f] else float(row[f])) for f in FEATURES}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_valid_probability(client):
    r = client.post("/predict", json={"features": _company(0)})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["flag"] == (body["probability"] >= body["threshold"])


def test_predict_accepts_partial_features_and_headlines(client):
    r = client.post("/predict", json={
        "features": {"X1": -0.2, "X2": 1.3, "X4": 0.5},
        "headlines": ["Rating agency downgrades company after covenant breach"]})
    assert r.status_code == 200
    assert r.json()["news_sentiment"]["label"] == "negative"


def test_predict_rejects_unknown_feature_and_empty_payload(client):
    assert client.post("/predict", json={"features": {"X999": 1.0}}).status_code == 422
    assert client.post("/predict", json={"features": {}}).status_code == 422


def test_sentiment_endpoint(client):
    r = client.post("/sentiment", json={"headlines": ["Profit surges as firm beats expectations"]})
    assert r.status_code == 200 and r.json()["summary"]["label"] == "positive"
