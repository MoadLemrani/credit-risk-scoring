"""Feature engineering on financial ratios (scikit-learn compatible transformers)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import FEATURES

# Interpretable distress flags added to the 64 raw ratios.
ENGINEERED = [
    "missing_count",             # how many ratios are unreported
    "neg_profit_flags",          # negative net profit / EBIT / gross profit / operating profit over assets
    "liabilities_exceed_assets",  # X2 (liabilities / assets) > 1  -> negative equity
    "current_ratio_below_1",     # X4 (current assets / short-term liabilities) < 1
]
ALL_FEATURE_NAMES = FEATURES + ENGINEERED


class FinancialFeatures(BaseEstimator, TransformerMixin):
    """Append distress flags computed from the raw ratios (NaN-safe: NaN comparisons are False)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=FEATURES) if not isinstance(X, pd.DataFrame) else X[FEATURES]
        out = X.copy()
        out["missing_count"] = X.isna().sum(axis=1)
        out["neg_profit_flags"] = (X[["X1", "X7", "X18", "X22"]] < 0).sum(axis=1)
        out["liabilities_exceed_assets"] = (X["X2"] > 1).astype(int)
        out["current_ratio_below_1"] = (X["X4"] < 1).astype(int)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(ALL_FEATURE_NAMES)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to its [lower, upper] training quantiles (financial ratios have heavy tails)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)
