import numpy as np
import pandas as pd

from creditrisk.config import FEATURES
from creditrisk.features import ALL_FEATURE_NAMES, ENGINEERED, FinancialFeatures, Winsorizer


def _frame(**overrides):
    row = {f: 0.5 for f in FEATURES}
    row.update(overrides)
    return pd.DataFrame([row])


def test_engineered_columns_added():
    out = FinancialFeatures().fit_transform(_frame())
    assert list(out.columns) == ALL_FEATURE_NAMES
    assert set(ENGINEERED) <= set(out.columns)


def test_distress_flags():
    healthy = FinancialFeatures().transform(_frame(X1=0.1, X2=0.4, X4=2.0))
    distressed = FinancialFeatures().transform(_frame(X1=-0.2, X7=-0.1, X2=1.4, X4=0.6))
    assert healthy.loc[0, "neg_profit_flags"] == 0
    assert healthy.loc[0, "liabilities_exceed_assets"] == 0
    assert distressed.loc[0, "neg_profit_flags"] == 2
    assert distressed.loc[0, "liabilities_exceed_assets"] == 1
    assert distressed.loc[0, "current_ratio_below_1"] == 1


def test_missing_values_counted_and_nan_safe():
    out = FinancialFeatures().transform(_frame(X2=np.nan, X4=np.nan))
    assert out.loc[0, "missing_count"] == 2
    assert out.loc[0, "liabilities_exceed_assets"] == 0  # NaN comparison is False, not an error


def test_winsorizer_clips_to_training_quantiles():
    X = np.arange(101, dtype=float).reshape(-1, 1)
    w = Winsorizer(0.05, 0.95).fit(X)
    out = w.transform(np.array([[-1000.0], [50.0], [1000.0]]))
    assert out[0, 0] == w.low_[0] and out[2, 0] == w.high_[0] and out[1, 0] == 50.0
