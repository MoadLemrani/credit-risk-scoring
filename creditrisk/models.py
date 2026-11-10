"""Candidate models. Every candidate shares the same preprocessing.

Class imbalance (~4.8 % bankrupt) is handled either with class weights
(logreg / random forest / xgboost / hist-gradient-boosting) or with SMOTE (xgb_smote ablation).
"""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import SEED
from .features import FinancialFeatures, Winsorizer


def _prep():
    return [
        ("features", FinancialFeatures()),
        ("impute", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer(0.01, 0.99)),
    ]


def build_models(pos_weight: float = 20.0, quick: bool = False) -> dict:
    """Return {name: Pipeline}. ``quick`` shrinks the ensembles (used by tests)."""
    n = 40 if quick else 400
    return {
        "logreg": Pipeline(_prep() + [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, C=0.1)),
        ]),
        "random_forest": Pipeline(_prep() + [
            ("clf", RandomForestClassifier(
                n_estimators=n // 2 if quick else 300, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=2, random_state=SEED)),
        ]),
        "xgboost": Pipeline(_prep() + [
            ("clf", XGBClassifier(
                n_estimators=n, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                scale_pos_weight=pos_weight, eval_metric="aucpr", n_jobs=2, random_state=SEED)),
        ]),
        "gradient_boosting": Pipeline(_prep() + [
            ("clf", HistGradientBoostingClassifier(
                max_iter=n, learning_rate=0.05, max_leaf_nodes=31, class_weight="balanced",
                random_state=SEED)),
        ]),
        # ablation: SMOTE oversampling instead of class weights
        "xgboost_smote": Pipeline(_prep() + [
            ("smote", SMOTE(sampling_strategy=0.3, random_state=SEED)),
            ("clf", XGBClassifier(
                n_estimators=n, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                eval_metric="aucpr", n_jobs=2, random_state=SEED)),
        ]),
    }
