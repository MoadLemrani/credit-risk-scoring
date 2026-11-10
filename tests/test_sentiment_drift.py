import numpy as np
import pandas as pd

from creditrisk import drift, sentiment


def test_sentiment_labels():
    pos = sentiment.score_headline("Company beats forecasts, raises dividend after record profit")
    neg = sentiment.score_headline("Rating agency downgrades firm after default on bond payment")
    neu = sentiment.score_headline("Company to hold annual meeting on Tuesday")
    assert pos["label"] == "positive"
    assert neg["label"] == "negative"
    assert neu["label"] == "neutral"


def test_sentiment_aggregate_empty():
    assert sentiment.aggregate([])["label"] == "neutral"


def test_psi_zero_for_identical_and_large_for_shift():
    rng = np.random.RandomState(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.normal(0, 1, 5000))
    shifted = pd.Series(rng.normal(2, 1, 5000))
    assert drift.psi(a, b) < 0.05
    assert drift.psi(a, shifted) > 0.25


def test_drift_summary_counts_drifted_features():
    rng = np.random.RandomState(1)
    ref = pd.DataFrame({"a": rng.normal(size=3000), "b": rng.normal(size=3000)})
    cur = pd.DataFrame({"a": rng.normal(size=3000), "b": rng.normal(3, 1, size=3000)})
    s = drift.drift_summary(ref, cur)
    assert s["n_psi_gt_0.25"] == 1 and "b" in s["top_drifted"]
