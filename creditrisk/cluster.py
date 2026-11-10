"""Unsupervised segmentation of companies into risk profiles (K-means on key ratios)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED, TARGET
from .features import Winsorizer

# Liquidity, leverage, profitability and activity ratios (names from data/FEATURES.txt)
CLUSTER_FEATURES = {
    "X1": "net profit / total assets",
    "X2": "total liabilities / total assets",
    "X3": "working capital / total assets",
    "X4": "current assets / short-term liabilities",
    "X6": "retained earnings / total assets",
    "X7": "EBIT / total assets",
    "X9": "sales / total assets",
    "X19": "gross profit / sales",
}
COLS = list(CLUSTER_FEATURES)


def _prep():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer(0.02, 0.98)),
        ("scale", StandardScaler()),
    ])


def select_k(df: pd.DataFrame, ks=range(3, 7), sample: int = 8000) -> dict:
    """Silhouette score for each k (on a fixed random subsample for speed)."""
    Z = _prep().fit_transform(df[COLS])
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(Z), size=min(sample, len(Z)), replace=False)
    scores = {}
    for k in ks:
        labels = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
        scores[k] = float(silhouette_score(Z[idx], labels[idx]))
    return scores


def fit_segments(df: pd.DataFrame, k: int):
    """Fit preprocessing + K-means; return (pipeline, kmeans, profile table)."""
    prep = _prep().fit(df[COLS])
    Z = prep.transform(df[COLS])
    km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
    labels = km.labels_
    prof = df[COLS].assign(cluster=labels, **{TARGET: df[TARGET].values}).groupby("cluster").agg(
        **{"n_companies": (TARGET, "size"), "bankruptcy_rate": (TARGET, "mean")},
        **{c: (c, "median") for c in COLS},
    )
    prof["share"] = prof["n_companies"] / prof["n_companies"].sum()
    return prep, km, prof.round(4)


def assign(prep, km, X: pd.DataFrame) -> np.ndarray:
    return km.predict(prep.transform(X[COLS]))
