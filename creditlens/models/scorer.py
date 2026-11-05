"""
Credit scoring models.

LogisticScorer (what the API serves), XGBoostScorer and NeuralNetScorer share one
interface: fit(), predict_proba(), predict_score() (0-1000 scale).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import ks_2samp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from creditlens.decision import probability_to_score

logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluation:
    """Comprehensive model evaluation results."""

    model_name: str
    auc_roc: float
    auc_pr: float
    brier_score: float
    ks_statistic: float
    gini_coefficient: float
    classification_report: str
    confusion_matrix: list[list[int]]
    threshold_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __repr__(self) -> str:
        return (
            f"ModelEvaluation({self.model_name} | "
            f"AUC-ROC={self.auc_roc:.3f} | "
            f"AUC-PR={self.auc_pr:.3f} | "
            f"Gini={self.gini_coefficient:.3f})"
        )


class LogisticScorer:
    """
    Sign-constrained logistic regression: the model the API serves.

    The synthetic default label is a logistic function of the model's features
    (data/generator.py ``default_log_odds``), so this is the correctly specified
    form, and it is the standard credit-scorecard form. Each feature's weight is
    bounded to its documented risk direction
    (``creditlens.features.engineering.RISK_DIRECTION``), so a worse credit signal
    can never lower the predicted risk: where the data would give an input the
    wrong sign, its weight is exactly 0. Inputs are standardised internally.

    It maximises the (optionally weighted) likelihood with L-BFGS-B from a zero
    start, so the same data always gives the same model. No regularisation.

    Parameters
    ----------
    monotone : bool
        Bound each weight named in ``RISK_DIRECTION`` to its direction. Default True.
    max_iter : int
        L-BFGS-B iteration limit. Not converging within it raises ``RuntimeError``.
    """

    def __init__(self, monotone: bool = True, max_iter: int = 2000) -> None:
        self.monotone = monotone
        self.max_iter = max_iter
        self._feature_names: list[str] = []
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._weights: np.ndarray | None = None  # per standardised feature
        self._intercept: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> "LogisticScorer":
        """Fit by weighted maximum likelihood. ``sample_weight`` is per row, e.g.
        from ``FairnessAuditor.mitigation_reweighting``."""
        names = list(X.columns)
        A = X.to_numpy(dtype=float)
        target = np.asarray(y, dtype=float)
        if sample_weight is None:
            weights = np.ones(len(A))
        else:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != (len(A),) or not np.all(np.isfinite(weights)) or (weights < 0).any():
                raise ValueError("sample_weight must be one finite, non-negative weight per row")
        weights = weights / weights.sum()

        mean = A.mean(axis=0)
        scale = A.std(axis=0)
        scale[scale == 0] = 1.0
        Z = np.column_stack([np.ones(len(A)), (A - mean) / scale])

        def loss(beta: np.ndarray) -> tuple[float, np.ndarray]:
            margin = Z @ beta
            value = np.sum(weights * (np.logaddexp(0.0, margin) - target * margin))
            return float(value), Z.T @ (weights * (expit(margin) - target))

        bounds: list[tuple[float | None, float | None]] = [(None, None)] * Z.shape[1]
        if self.monotone:
            from creditlens.features.engineering import RISK_DIRECTION
            side = {-1: (None, 0.0), 1: (0.0, None)}
            bounds = [(None, None)] + [side.get(RISK_DIRECTION.get(n, 0), (None, None)) for n in names]

        result = minimize(loss, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": self.max_iter, "maxfun": 10 * self.max_iter,
                                   "ftol": 1e-13, "gtol": 1e-7})
        if not result.success:
            raise RuntimeError(f"LogisticScorer did not converge: {result.message}")

        self._feature_names = names
        self._mean, self._scale = mean, scale
        self._intercept, self._weights = float(result.x[0]), result.x[1:]
        logger.info(f"LogisticScorer trained on {len(A)} samples.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted default probabilities. Shape: (n, 2)."""
        if self._weights is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        A = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        p = expit(self._intercept + ((A - self._mean) / self._scale) @ self._weights)
        return np.column_stack([1.0 - p, p])

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Credit scores on the 0-1000 scale; higher = lower default risk."""
        return probability_to_score(self.predict_proba(X)[:, 1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions at given probability threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def coefficients(self) -> pd.Series:
        """Log-odds of default per one unit of each column of ``X``."""
        if self._weights is None:
            return pd.Series(dtype=float)
        return pd.Series(self._weights / self._scale, index=self._feature_names)

    def intercept(self) -> float:
        """Log-odds of default when every column of ``X`` is 0."""
        return float(self._intercept - np.sum(self._weights * self._mean / self._scale))

    def feature_importances(self) -> pd.Series:
        """Absolute weight per standardised feature, largest first."""
        if self._weights is None:
            logger.warning("feature_importances() called on an untrained model; returning empty.")
            return pd.Series(dtype=float)
        return pd.Series(np.abs(self._weights), index=self._feature_names).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        import joblib
        joblib.dump({"features": self._feature_names, "mean": self._mean, "scale": self._scale,
                     "intercept": self._intercept, "weights": self._weights}, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> "LogisticScorer":
        """Load model from disk."""
        import joblib
        state = joblib.load(path)
        self._feature_names, self._mean, self._scale = state["features"], state["mean"], state["scale"]
        self._intercept, self._weights = state["intercept"], state["weights"]
        return self

    def __repr__(self) -> str:
        fitted = "fitted" if self._weights is not None else "unfitted"
        return f"LogisticScorer(monotone={self.monotone}, {fitted})"


class XGBoostScorer:
    """
    XGBoost-based credit default predictor.

    Gradient boosted trees, kept for library use and for comparison (README
    "Why logistic regression"). The API does not serve it: a tree's response is a
    step function, so even depth-1 trees lump an additive effect at split points.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds (default 300).
    max_depth : int
        Max tree depth (default 5).
    learning_rate : float
        Boosting learning rate (default 0.05).
    scale_pos_weight : float | None
        Class imbalance correction. If None, auto-computes from data.
    calibrate : bool
        Apply isotonic calibration (on cross-validated held-out folds) for
        reliable, well-spread probability estimates. When True, scale_pos_weight
        rebalancing is skipped so the calibrated scores keep a wide range.
    monotone : bool
        Constrain every feature listed in
        ``creditlens.features.engineering.RISK_DIRECTION`` to move predicted
        P(default) only in its documented direction (XGBoost
        ``monotone_constraints``). Default True. Isotonic calibration is
        monotone and the fold ensemble is a mean, so the served probability
        keeps the guarantee.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float | None = None,
        calibrate: bool = True,
        monotone: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.calibrate = calibrate
        self.monotone = monotone
        self._model = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> "XGBoostScorer":
        """
        Train the XGBoost model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Binary labels (0 = no default, 1 = default).
        sample_weight : array-like, optional
            Per-row training weights, e.g. from
            ``FairnessAuditor.mitigation_reweighting``. Used by every booster
            and, when calibrating, by the isotonic calibrators. ``None`` (the
            default) trains exactly as before.

        Returns
        -------
        self
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        self._feature_names = list(X.columns)

        # Class-imbalance handling.
        #
        # scale_pos_weight rescales the positive class during training. That is
        # useful for an *uncalibrated* model, but stacking Platt/sigmoid
        # calibration on top of a scale_pos_weight-rebalanced booster squeezes
        # the calibrated P(default) into a narrow band — every applicant then
        # scores in a tiny window and no one can be denied. So when calibrating
        # we deliberately do NOT apply scale_pos_weight (unless the caller set it
        # explicitly) and instead let isotonic calibration recover well-spread,
        # reliable probabilities from a held-out calibration set.
        if self.scale_pos_weight is not None:
            spw = self.scale_pos_weight
        elif self.calibrate:
            spw = 1.0  # no rebalancing under calibration -> wide, honest score range
        else:
            n_neg = (y == 0).sum()
            n_pos = (y == 1).sum()
            spw = max(n_neg / max(n_pos, 1), 1.0)

        constraints = {}
        if self.monotone:
            from creditlens.features.engineering import RISK_DIRECTION
            constraints["monotone_constraints"] = tuple(
                RISK_DIRECTION.get(c, 0) for c in self._feature_names
            )

        base_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=spw,
            random_state=42,
            eval_metric="aucpr",
            verbosity=0,
            **constraints,
        )

        if self.calibrate:
            # Isotonic calibration on cross-validated held-out folds. Isotonic is
            # more flexible than sigmoid and, combined with dropping
            # scale_pos_weight above, yields calibrated probabilities that span a
            # wide range so all three decision tiers stay reachable.
            self._model = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
        else:
            self._model = base_model

        if sample_weight is None:
            self._model.fit(X.values, y.values)
        else:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != (len(X),) or not np.all(np.isfinite(weights)) or (weights < 0).any():
                raise ValueError("sample_weight must be one finite, non-negative weight per row")
            self._model.fit(X.values, y.values, sample_weight=weights)
        logger.info(f"XGBoostScorer trained on {len(X)} samples.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted default probabilities. Shape: (n, 2)."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._model.predict_proba(X.values)

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return credit scores on 0-1000 scale.

        Higher score = lower default risk (industry convention). Same mapping
        as the API (creditlens.decision.probability_to_score).
        """
        return probability_to_score(self.predict_proba(X)[:, 1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions at given probability threshold."""
        proba_default = self.predict_proba(X)[:, 1]
        return (proba_default >= threshold).astype(int)

    def feature_importances(self) -> pd.Series:
        """
        Return feature importances.

        For calibrated model, averages across the per-fold fitted estimators.
        """
        if self._model is None:
            logger.warning("feature_importances() called on an untrained model; returning empty.")
            return pd.Series(dtype=float)

        importances: np.ndarray | None = None

        try:
            if self.calibrate and hasattr(self._model, "calibrated_classifiers_"):
                # Average importances across calibrated folds
                arrays = []
                for cc in self._model.calibrated_classifiers_:
                    est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
                    if est is not None and hasattr(est, "feature_importances_"):
                        arrays.append(np.asarray(est.feature_importances_))
                if arrays:
                    importances = np.mean(arrays, axis=0)
            elif not self.calibrate and hasattr(self._model, "feature_importances_"):
                importances = np.asarray(self._model.feature_importances_)
        except Exception as e:
            # Do NOT swallow silently: an empty Series returned here is
            # indistinguishable from "model has no important features" and would
            # quietly hide a broken importance-extraction path. Log the real
            # cause; still return the empty-Series contract callers expect.
            logger.warning("feature_importances() extraction failed: %s; returning empty.", e)
            importances = None

        if importances is None or len(importances) == 0:
            if importances is not None:
                logger.warning(
                    "feature_importances() found no per-fold importances; returning empty."
                )
            return pd.Series(dtype=float)

        # Align length with feature names (defensive)
        names = self._feature_names[: len(importances)]
        return (
            pd.Series(importances[: len(names)], index=names)
            .sort_values(ascending=False)
        )

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        import joblib
        joblib.dump(self._model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> "XGBoostScorer":
        """Load model from disk."""
        import joblib
        self._model = joblib.load(path)
        return self

    def __repr__(self) -> str:
        fitted = "fitted" if self._model is not None else "unfitted"
        return f"XGBoostScorer(n_estimators={self.n_estimators}, {fitted})"


class NeuralNetScorer:
    """
    Multi-layer perceptron credit scorer.

    Another model with the same interface, for comparing performance. The API
    does not use it, and the library contains no ensembling code.

    Parameters
    ----------
    hidden_layers : tuple[int, ...]
        Hidden layer sizes (default (256, 128, 64)).
    dropout : float
        Dropout rate (handled via sklearn alpha regularization).
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (256, 128, 64),
        max_iter: int = 200,
        learning_rate_init: float = 0.001,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self._model: MLPClassifier | None = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NeuralNetScorer":
        self._feature_names = list(X.columns)
        self._model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            activation="relu",
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False,
        )
        self._model.fit(X.values, y.values)
        logger.info(f"NeuralNetScorer trained on {len(X)} samples.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained.")
        return self._model.predict_proba(X.values)

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        return probability_to_score(self.predict_proba(X)[:, 1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def __repr__(self) -> str:
        return f"NeuralNetScorer(layers={self.hidden_layers})"


def evaluate_model(
    model: LogisticScorer | XGBoostScorer | NeuralNetScorer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "CreditModel",
) -> ModelEvaluation:
    """
    Comprehensive model evaluation.

    Parameters
    ----------
    model : LogisticScorer | XGBoostScorer | NeuralNetScorer
        Trained model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels.
    model_name : str
        Label for the evaluation output.

    Returns
    -------
    ModelEvaluation
    """
    y_true = y_test.values
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    gini = 2 * auc_roc - 1

    # Kolmogorov-Smirnov statistic: the maximum vertical gap between the
    # cumulative distributions of predicted probabilities for defaulters vs.
    # non-defaulters, KS = max|CDF_pos(t) - CDF_neg(t)|. scipy's two-sample KS
    # computes exactly this (the previous searchsorted construction did not).
    pos_scores = y_proba[y_true == 1]
    neg_scores = y_proba[y_true == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        ks_stat = float(ks_2samp(pos_scores, neg_scores).statistic)
    else:
        ks_stat = 0.0

    # Threshold analysis
    thresholds = np.arange(0.1, 0.9, 0.1)
    threshold_rows = []
    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        threshold_rows.append({
            "threshold": thresh,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "approval_rate": (preds == 0).sum() / len(preds),
        })

    cm = confusion_matrix(y_true, y_pred).tolist()

    return ModelEvaluation(
        model_name=model_name,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        brier_score=brier,
        ks_statistic=ks_stat,
        gini_coefficient=gini,
        classification_report=classification_report(y_true, y_pred),
        confusion_matrix=cm,
        threshold_metrics=pd.DataFrame(threshold_rows),
    )
