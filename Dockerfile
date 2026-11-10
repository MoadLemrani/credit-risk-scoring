# Python version matches the one used to train (model.joblib is a pickle: keep versions aligned).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CREDITRISK_ARTIFACTS=/app/artifacts
WORKDIR /app

# xgboost needs libgomp at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY creditrisk/ creditrisk/
# trained artifacts: run `python -m creditrisk.train` before building
COPY artifacts/model.joblib artifacts/segments.joblib artifacts/cluster_profile.csv artifacts/metrics.json artifacts/

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "creditrisk.api:app", "--host", "0.0.0.0", "--port", "8000"]
