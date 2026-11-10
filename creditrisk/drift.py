"""Data-drift monitoring: PSI (lightweight, testable) + an Evidently HTML report."""
from pathlib import Path

import numpy as np
import pandas as pd


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with quantile bins from the reference.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
    """
    ref = reference.dropna().to_numpy()
    cur = current.dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.Series:
    return pd.Series({c: psi(reference[c], current[c]) for c in reference.columns}).sort_values(ascending=False)


def drift_summary(reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.25) -> dict:
    t = psi_table(reference, current)
    return {
        "n_features": int(len(t)),
        "n_psi_gt_0.1": int((t > 0.1).sum()),
        "n_psi_gt_0.25": int((t > threshold).sum()),
        "max_psi": float(t.max()),
        "top_drifted": {k: round(float(v), 3) for k, v in t.head(5).items()},
    }


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, html_path: Path) -> dict:
    """Run Evidently's DataDriftPreset and save the HTML report. Returns drifted-column count/share."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    # Financial ratios are extremely heavy-tailed: on raw values Evidently's default tests flag "drift"
    # even between two random halves of the same data. Clip both sets to the reference 1-99 % range first.
    lo, hi = reference.quantile(0.01), reference.quantile(0.99)
    reference, current = reference.clip(lo, hi, axis=1), current.clip(lo, hi, axis=1)

    snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    first = snapshot.dict()["metrics"][0]["value"]
    return {"drifted_columns": int(first["count"]), "drifted_share": float(first["share"])}
