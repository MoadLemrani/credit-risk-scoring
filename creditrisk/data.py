"""Data loading."""
import pandas as pd

from .config import DATA_PATH, FEATURES, TARGET


def load_data(path=DATA_PATH, drop_duplicates: bool = True) -> pd.DataFrame:
    """Load the pooled Polish-companies bankruptcy data (5 forecasting horizons).

    Columns: ``year`` (horizon file, '1year'..'5year'), X1..X64 (financial ratios),
    ``bankrupt`` (1 = bankrupt within the forecasting horizon).
    Exact duplicate feature rows are dropped (they would leak across CV folds).
    """
    df = pd.read_csv(path)
    if drop_duplicates:
        df = df.drop_duplicates(subset=FEATURES).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    return df
