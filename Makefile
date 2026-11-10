.PHONY: install train test serve docker mlflow
install:
	pip install -r requirements.txt
train:
	python -m creditrisk.train
test:
	pytest -q
serve:
	uvicorn creditrisk.api:app --reload
docker:
	docker build -t credit-risk-api . && docker run -p 8000:8000 credit-risk-api
mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
