"""End-to-end training: CV model comparison -> held-out test -> clustering -> drift -> artifacts.

Run:  python -m creditrisk.train            (full run, ~10 min on 2 CPUs)
      python -m creditrisk.train --quick    (tiny models + 3k rows, for smoke tests)
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,  # noqa: E402
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

from . import cluster as seg  # noqa: E402
from . import drift  # noqa: E402
from .config import (ARTIFACTS, FEATURES, MLFLOW_EXPERIMENT, MLFLOW_URI, SEED, TARGET)  # noqa: E402
from .data import load_data  # noqa: E402
from .features import ALL_FEATURE_NAMES  # noqa: E402
from .models import build_models  # noqa: E402


def f2_threshold(y, p) -> float:
    """Decision threshold maximising F2 (recall weighted 2x precision: missing a default costs more)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f2 = 5 * prec[:-1] * rec[:-1] / np.clip(4 * prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f2)])


def metrics_at(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "threshold": float(thr),
        "recall": float(recall_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def cross_validate_models(models: dict, X, y, n_splits: int = 5) -> tuple[pd.DataFrame, dict]:
    """Stratified K-fold; returns summary table and out-of-fold probabilities per model."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows, oof_all = [], {}
    for name, model in models.items():
        t0 = time.time()
        oof = np.zeros(len(y))
        aucs, aps = [], []
        for tr, va in skf.split(X, y):
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict_proba(X.iloc[va])[:, 1]
            oof[va] = p
            aucs.append(roc_auc_score(y.iloc[va], p))
            aps.append(average_precision_score(y.iloc[va], p))
        thr = f2_threshold(y, oof)
        pred = oof >= thr
        rows.append({
            "model": name,
            "cv_roc_auc_mean": np.mean(aucs), "cv_roc_auc_std": np.std(aucs),
            "cv_pr_auc_mean": np.mean(aps),
            "oof_threshold_f2": thr,
            "oof_recall": recall_score(y, pred), "oof_precision": precision_score(y, pred, zero_division=0),
            "seconds": time.time() - t0,
        })
        oof_all[name] = oof
        print(f"  {name:18s} AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  PR-AUC {np.mean(aps):.4f}  "
              f"recall@F2 {rows[-1]['oof_recall']:.3f}  ({rows[-1]['seconds']:.0f}s)", flush=True)
    return pd.DataFrame(rows).set_index("model"), oof_all


def plot_curves(y, p, path: Path, title: str):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, lw=2)
    ax[0].plot([0, 1], [0, 1], "--", c="grey")
    ax[0].set(xlabel="False positive rate", ylabel="Recall", title=f"ROC (AUC {roc_auc_score(y, p):.3f})")
    prec, rec, _ = precision_recall_curve(y, p)
    ax[1].plot(rec, prec, lw=2)
    ax[1].axhline(y.mean(), ls="--", c="grey")
    ax[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall (AP {average_precision_score(y, p):.3f})")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(pipe, path: Path, top: int = 15):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=ALL_FEATURE_NAMES)
    elif hasattr(clf, "coef_"):
        imp = pd.Series(np.abs(clf.coef_[0]), index=ALL_FEATURE_NAMES)
    else:
        return None
    imp = imp.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    imp.head(top)[::-1].plot.barh(ax=ax)
    ax.set_title(f"Top {top} features")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return imp.head(top).round(4).to_dict()


def run(outdir: Path = ARTIFACTS, quick: bool = False, use_mlflow: bool = True) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_data()
    n_raw_dups = 43400 - len(df) if not quick else 0
    if quick:
        df = df.groupby(TARGET, group_keys=False).sample(frac=3000 / len(df), random_state=SEED).reset_index(drop=True)
    X, y = df[FEATURES], df[TARGET]
    print(f"Data: {len(df):,} companies, {len(FEATURES)} ratios, bankrupt rate {y.mean():.2%}", flush=True)

    X_tr, X_te, y_tr, y_te, yr_tr, yr_te = train_test_split(
        X, y, df["year"], test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    models = build_models(pos_weight, quick=quick)

    print("Stratified 5-fold CV on the training split:", flush=True)
    cv, oof = cross_validate_models(models, X_tr, y_tr)
    best = cv["cv_roc_auc_mean"].idxmax()
    thr = float(cv.loc[best, "oof_threshold_f2"])
    print(f"Best by CV ROC-AUC: {best} (decision threshold {thr:.3f})", flush=True)

    final = clone(models[best]).fit(X_tr, y_tr)
    p_te = final.predict_proba(X_te)[:, 1]
    test = metrics_at(y_te, p_te, thr)
    print("Held-out test:", {k: round(v, 4) if isinstance(v, float) else v for k, v in test.items()}, flush=True)

    # robustness: does it transfer to a different forecasting horizon? (train years 1-4 -> test year 5)
    mask5 = df["year"] == "5year"
    h = clone(models[best]).fit(X[~mask5], y[~mask5])
    p5 = h.predict_proba(X[mask5])[:, 1]
    horizon = metrics_at(y[mask5], p5, thr)
    horizon["roc_auc"] = float(roc_auc_score(y[mask5], p5))
    print(f"Cross-horizon (train 1-4year, test 5year): AUC {horizon['roc_auc']:.4f}", flush=True)

    # final production model: refit on ALL data with the chosen threshold
    prod = clone(models[best]).fit(X, y)

    # --- segmentation -------------------------------------------------------------------------
    ks = seg.select_k(df)
    k = max(ks, key=ks.get)
    prep, km, profile = seg.fit_segments(df, k)
    print(f"K-means silhouette by k: { {a: round(b, 3) for a, b in ks.items()} } -> k={k}", flush=True)

    # --- drift --------------------------------------------------------------------------------
    drift_split = drift.drift_summary(X_tr, X_te)
    drift_horizon = drift.drift_summary(X[~mask5], X[mask5])
    ev = {}
    if not quick:
        ev["train_vs_test"] = drift.evidently_report(
            X_tr.sample(5000, random_state=SEED), X_te.sample(5000, random_state=SEED), outdir / "drift_train_vs_test.html")
        ev["year1-4_vs_year5"] = drift.evidently_report(
            X[~mask5].sample(5000, random_state=SEED), X[mask5].sample(5000, random_state=SEED),
            outdir / "drift_horizon_shift.html")

    # --- artifacts ----------------------------------------------------------------------------
    importances = plot_importance(final, outdir / "feature_importance.png")
    plot_curves(y_te, p_te, outdir / "roc_pr_test.png", f"{best} - held-out test (n={len(y_te):,})")
    joblib.dump(prod, outdir / "model.joblib")
    joblib.dump({"prep": prep, "kmeans": km}, outdir / "segments.joblib")
    profile.to_csv(outdir / "cluster_profile.csv")
    cv.round(4).to_csv(outdir / "cv_results.csv")
    meta = {
        "best_model": best, "threshold": thr, "n_companies": int(len(df)), "n_features": len(FEATURES),
        "bankrupt_rate": float(y.mean()), "duplicates_dropped": int(n_raw_dups),
        "test": test, "cross_horizon_year5": horizon,
        "kmeans": {"k": int(k), "silhouette_by_k": {int(a): b for a, b in ks.items()}},
        "drift_psi": {"train_vs_test": drift_split, "year1-4_vs_year5": drift_horizon},
        "drift_evidently": ev, "top_features": importances,
    }
    (outdir / "metrics.json").write_text(json.dumps(meta, indent=2))

    if use_mlflow:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        for name, row in cv.iterrows():
            with mlflow.start_run(run_name=f"cv-{name}"):
                mlflow.log_param("model", name)
                mlflow.log_metrics({k_: float(v) for k_, v in row.items()})
        with mlflow.start_run(run_name=f"final-{best}"):
            mlflow.log_params({"model": best, "threshold": round(thr, 4), "seed": SEED, "pos_weight": round(pos_weight, 2)})
            mlflow.log_metrics({f"test_{k_}": v for k_, v in test.items() if isinstance(v, float)})
            mlflow.log_metric("cross_horizon_roc_auc", horizon["roc_auc"])
            mlflow.log_artifacts(str(outdir))
    print(f"Artifacts written to {outdir}", flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny models on a 3k-row sample (smoke test)")
    ap.add_argument("--out", default=str(ARTIFACTS))
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    run(Path(a.out), quick=a.quick, use_mlflow=not a.no_mlflow)
