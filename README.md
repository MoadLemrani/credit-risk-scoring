# Corporate Credit-Risk Scoring

End-to-end ML pipeline that predicts whether a company will go bankrupt from 64 financial ratios, segments companies into risk profiles, scores financial-news sentiment, and serves everything through a FastAPI service with experiment tracking, drift monitoring, tests and CI.

**Data:** [Polish companies bankruptcy data](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data) (UCI, CC BY 4.0; Zięba, Tomczak & Tomczak, 2016). 5 forecasting horizons pooled: 43,400 company-observations, 401 exact duplicates removed → **42,999 rows, 64 ratios, 4.84 % bankrupt** (heavy class imbalance). Feature definitions: `data/FEATURES.txt`.

## Results (all numbers are produced by `python -m creditrisk.train`; see `artifacts/metrics.json`)

80/20 stratified split. Model selection and threshold choice use **only the 80 % training split** (stratified 5-fold CV); the 20 % test split is touched once.

| Model | CV ROC-AUC (mean ± std) | CV PR-AUC | Recall @ F2 threshold (OOF) |
|---|---|---|---|
| Logistic regression (class weights) | 0.812 ± 0.008 | 0.179 | 0.64 |
| Random forest (class weights) | 0.942 ± 0.005 | 0.623 | 0.75 |
| **XGBoost (class weights)** | **0.966 ± 0.002** | **0.784** | **0.78** |
| Gradient boosting (hist-GB, class weights) | 0.962 ± 0.003 | 0.750 | 0.78 |
| XGBoost + SMOTE (ablation) | 0.961 ± 0.005 | 0.763 | 0.74 |

**Held-out test (XGBoost, n = 8,600, 417 bankruptcies):** ROC-AUC **0.961**, PR-AUC 0.760, at the F2-optimal threshold (0.444): **recall 0.770**, precision 0.514 (321 of 417 bankruptcies caught, 304 false alarms), Brier 0.032.

Class weights ≈ SMOTE here (0.966 vs 0.961 AUC), so the simpler option is kept.

**Segmentation (K-means on 8 liquidity / leverage / profitability / activity ratios, k = 3 by silhouette 0.28):**

| Profile | Share | Bankruptcy rate | Description |
|---|---|---|---|
| 0 | 11 % | **13.5 %** | loss-making, liabilities ≈ 91 % of assets, current ratio 0.77 |
| 1 | 27 % | 1.9 % | profitable, liabilities ≈ 21 % of assets, current ratio 3.8 |
| 2 | 62 % | 4.6 % | middle of the pack |

**Drift monitoring:** PSI (per feature) and an Evidently report (`artifacts/drift_*.html`). Train vs. test (random split): 0/64 features drifted, as expected. Years 1–4 vs. year 5: 6/64 features drifted (max PSI 0.14, `X21` sales growth).

**News sentiment:** lexicon-based (VADER extended with 33 finance / credit-risk terms such as *default, downgrade, covenant breach*). No labeled data was used, so no accuracy is claimed; it is exposed as a separate signal next to the model probability and does not feed the model.

## Honest caveats

- **Optimistic estimate.** The public data has no company id, and the same firm may appear in several horizon files. A random split can therefore put near-identical observations of one firm in train and test. The cross-horizon check (train on 1–4 year files, test on 5-year file → AUC 0.973) does not remove that risk. Treat 0.96 as an upper bound for a genuinely new company.
- `missing_count` (how many ratios are unreported) is one of the top features. Missingness may reflect distress, or may be an artifact of how the data was collected.
- Fixed hyper-parameters, no tuning; probabilities are not calibrated beyond class weighting (Brier is reported, no calibration step).
- The production model served by the API is refit on all rows; the test numbers above are for the train-only model.

## Project layout

```
creditrisk/   config, data, features (distress flags + winsorizer), models, train, cluster, sentiment, drift, api
tests/        13 tests: features, sentiment, PSI, API end to end (trains a small model on a 3k sample)
artifacts/    model.joblib, segments.joblib, metrics.json, cv_results.csv, plots, Evidently HTML reports
.github/workflows/ci.yml   pytest, then train + docker build + container smoke test
```

## Run it

```bash
pip install -r requirements.txt
python -m creditrisk.train        # ~5 min on 2 CPUs; logs to MLflow (sqlite) and writes artifacts/
pytest -q                         # ~40 s
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn creditrisk.api:app        # http://localhost:8000/docs
docker build -t credit-risk-api . && docker run -p 8000:8000 credit-risk-api
```

```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "features": {"X1": -0.2, "X2": 1.3, "X4": 0.5},
  "headlines": ["Rating agency downgrades firm after covenant breach"]}'
```

Returns the bankruptcy probability, a high/medium/low band, the K-means risk profile with its historical bankruptcy rate, and aggregated headline sentiment. Missing ratios are imputed (median). Unknown feature names are rejected with HTTP 422.

## Verification status

| Item | Status |
|---|---|
| Training, CV, test metrics, clustering, drift, MLflow logging | run, results above |
| 13 pytest tests | passing |
| API in a clean environment with only `requirements-api.txt` | served and queried |
| `Dockerfile` | written, **not built** (no Docker daemon where this was developed): run `docker build` once |
| GitHub Actions workflow | written, **not run**: it runs when the repo is pushed to GitHub |
