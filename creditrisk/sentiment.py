"""Financial-news sentiment: VADER lexicon extended with finance / credit-risk terms.

This is a lexicon-based classifier (no training data, no claim of benchmark accuracy).
"""
from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Domain terms VADER's general-purpose lexicon does not know (scores on VADER's -4..+4 scale).
FINANCE_LEXICON = {
    "bankruptcy": -3.5, "insolvency": -3.5, "insolvent": -3.5, "default": -3.0, "defaults": -3.0,
    "defaulted": -3.0, "downgrade": -2.5, "downgraded": -2.5, "restructuring": -1.5,
    "delisting": -3.0, "lawsuit": -1.8, "fraud": -3.5, "layoffs": -2.0, "writedown": -2.5,
    "impairment": -2.0, "shortfall": -2.0, "covenant": -0.5, "breach": -2.5, "writeoff": -2.5,
    "upgrade": 2.5, "upgraded": 2.5, "outperform": 2.5, "beat": 1.8, "beats": 1.8, "surge": 2.0,
    "surges": 2.0, "rebound": 1.8, "dividend": 1.2, "profitable": 2.2, "profit": 1.5,
    "expansion": 1.5, "refinanced": 1.2, "deleveraging": 1.2,
}


@lru_cache(maxsize=1)
def _analyzer() -> SentimentIntensityAnalyzer:
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(FINANCE_LEXICON)
    return sia


def score_headline(text: str, threshold: float = 0.05) -> dict:
    compound = _analyzer().polarity_scores(text)["compound"]
    label = "positive" if compound >= threshold else "negative" if compound <= -threshold else "neutral"
    return {"headline": text, "compound": round(compound, 4), "label": label}


def score_headlines(texts: list[str]) -> list[dict]:
    return [score_headline(t) for t in texts]


def aggregate(scores: list[dict]) -> dict:
    """Average sentiment across a company's recent headlines."""
    if not scores:
        return {"n": 0, "mean_compound": 0.0, "label": "neutral"}
    mean = sum(s["compound"] for s in scores) / len(scores)
    label = "positive" if mean >= 0.05 else "negative" if mean <= -0.05 else "neutral"
    return {"n": len(scores), "mean_compound": round(mean, 4), "label": label}
