"""FastAPI service: bankruptcy probability, risk profile and news sentiment.

Run:  uvicorn creditrisk.api:app --reload
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from . import cluster as seg
from . import sentiment
from .config import ARTIFACTS, FEATURES

STATE: dict = {}


def load_artifacts(directory: Path | None = None) -> dict:
    d = Path(directory or os.environ.get("CREDITRISK_ARTIFACTS", ARTIFACTS))
    meta = json.loads((d / "metrics.json").read_text())
    segments = joblib.load(d / "segments.joblib")
    return {
        "model": joblib.load(d / "model.joblib"),
        "prep": segments["prep"], "kmeans": segments["kmeans"],
        "threshold": meta["threshold"], "model_name": meta["best_model"],
        "profile": pd.read_csv(d / "cluster_profile.csv", index_col=0),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.update(load_artifacts())
    yield
    STATE.clear()


app = FastAPI(title="Corporate credit-risk scoring", version="1.0.0", lifespan=lifespan)


class CompanyIn(BaseModel):
    features: dict[str, float | None] = Field(
        ..., description="Financial ratios keyed X1..X64 (see data/FEATURES.txt). Missing ratios may be omitted or null.")
    headlines: list[str] = Field(default_factory=list, description="Optional recent news headlines about the company.")

    @field_validator("features")
    @classmethod
    def known_features(cls, v):
        unknown = sorted(set(v) - set(FEATURES))
        if unknown:
            raise ValueError(f"unknown feature(s): {unknown[:5]}")
        return v


class PredictionOut(BaseModel):
    probability: float
    flag: bool
    risk_band: str
    threshold: float
    risk_profile: int
    profile_historical_bankruptcy_rate: float
    news_sentiment: dict | None = None


class HeadlinesIn(BaseModel):
    headlines: list[str] = Field(..., min_length=1, max_length=100)


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE.get("model_name"), "threshold": STATE.get("threshold")}


@app.post("/predict", response_model=PredictionOut)
def predict(company: CompanyIn):
    if not company.features:
        raise HTTPException(422, "at least one financial ratio is required")
    row = pd.DataFrame([{f: company.features.get(f) for f in FEATURES}], columns=FEATURES).astype(float)
    p = float(STATE["model"].predict_proba(row)[0, 1])
    thr = STATE["threshold"]
    band = "high" if p >= thr else "medium" if p >= thr / 2 else "low"
    cluster = int(seg.assign(STATE["prep"], STATE["kmeans"], row)[0])
    out = PredictionOut(
        probability=round(p, 5), flag=p >= thr, risk_band=band, threshold=round(thr, 4),
        risk_profile=cluster,
        profile_historical_bankruptcy_rate=float(STATE["profile"].loc[cluster, "bankruptcy_rate"]),
    )
    if company.headlines:
        out.news_sentiment = sentiment.aggregate(sentiment.score_headlines(company.headlines))
    return out


@app.post("/sentiment")
def news_sentiment(body: HeadlinesIn):
    scores = sentiment.score_headlines(body.headlines)
    return {"headlines": scores, "summary": sentiment.aggregate(scores)}
