"""Shared modeling helpers: preprocessing, candidate models, search spaces.

Kept separate so train.py and evaluate.py agree on how features are encoded.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    # sklearn >= 1.6 uses __sklearn_tags__ for is_classifier(); xgboost < 2.1 doesn't
    # provide it -> "Got a regressor with response_method=predict_proba". Add a
    # minimal shim (best-effort; if the Tags API differs, the pipeline still
    # gracefully skips this candidate via train.py's try/except).
    if not hasattr(XGBClassifier, "__sklearn_tags__"):
        try:                                                            # noqa: SIM105
            from sklearn.utils._tags import (
                ClassifierTags, InputTags, Tags, TargetTags,
            )

            def _xgb_tags(self):
                return Tags(
                    estimator_type="classifier",
                    classifier_tags=ClassifierTags(),
                    target_tags=TargetTags(required=True),
                    input_tags=InputTags(),
                )
            XGBClassifier.__sklearn_tags__ = _xgb_tags
        except Exception:
            pass
except Exception:  # pragma: no cover
    HAS_XGB = False

TREE_MODELS = {"random_forest", "xgboost"}


def make_preprocessor(numeric, categorical):
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        [("num", num, numeric), ("cat", cat, categorical)],
        remainder="drop", verbose_feature_names_out=True,
    )


def candidate(name, class_weight, scale_pos_weight, seed):
    """Return (estimator, param_distributions) for a model name."""
    if name == "logistic_l2":
        est = LogisticRegression(penalty="l2", max_iter=2000,
                                 class_weight=class_weight, random_state=seed)
        params = {"clf__C": loguniform(1e-3, 1e2)}
    elif name == "random_forest":
        est = RandomForestClassifier(class_weight=class_weight, random_state=seed, n_jobs=-1)
        params = {"clf__n_estimators": randint(100, 600),
                  "clf__max_depth": randint(2, 12),
                  "clf__min_samples_leaf": randint(1, 20)}
    elif name == "xgboost":
        if not HAS_XGB:
            return None, None
        est = XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=-1,
                            scale_pos_weight=scale_pos_weight or 1.0, tree_method="hist")
        params = {"clf__n_estimators": randint(100, 500),
                  "clf__max_depth": randint(2, 8),
                  "clf__learning_rate": uniform(0.01, 0.3),
                  "clf__subsample": uniform(0.6, 0.4),
                  "clf__colsample_bytree": uniform(0.6, 0.4)}
    else:
        raise ValueError(f"unknown candidate {name!r}")
    return est, params


def pos_weight(y):
    n_pos = int(np.sum(y))
    n_neg = len(y) - n_pos
    return (n_neg / n_pos) if n_pos else 1.0
