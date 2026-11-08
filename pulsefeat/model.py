"""Models evaluated with subject-wise cross-validation.

Windows from the same person look alike, so a random split leaks: the model sees the
same person in training and testing and the error looks far better than it really is.
GroupKFold keeps every subject entirely in one fold.
"""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


def models():
    return {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "gradient_boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                           random_state=0),
    }


def evaluate(df, target="sbp", n_splits=5):
    """Mean absolute error (mmHg) per model, averaged over subject-wise folds."""
    X, y, g = df[FEATURES].to_numpy(), df[target].to_numpy(), df["subject"].to_numpy()
    n_splits = min(n_splits, len(np.unique(g)))
    results = {}
    for name, m in models().items():
        maes = []
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
            m.fit(X[tr], y[tr])
            maes.append(np.mean(np.abs(m.predict(X[te]) - y[te])))
        results[name] = (float(np.mean(maes)), float(np.std(maes)))
    return results


def ridge_coefficients(df, target="sbp"):
    """Standardized ridge coefficients: which features move with pressure, and which way."""
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    m.fit(df[FEATURES].to_numpy(), df[target].to_numpy())
    return dict(zip(FEATURES, m[-1].coef_))
