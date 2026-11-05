"""
SHAP-based explainability for credit decisions.

Provides individual explanation (why was this applicant denied/approved?)
and global feature importance analysis.

What is computed
----------------
**Interventional Shapley values of the SERVED P(default)** — the probability
``/score`` returns — relative to a fixed background of real training
applicants. For applicant ``x`` and background ``R`` (1,000 of the rows passed to
``fit()``, every k-th by served probability, so its mean is the population's; the
API passes all its training rows), player ``j`` gets

    phi_j = E_{r in R} E_pi [ p(x_{S+j}, r_rest) - p(x_S, r_rest) ]

the average change in the served probability when ``j`` switches from the
background applicant's value to ``x``'s, over orderings ``pi`` of the players
(``S`` = the players before ``j`` in ``pi``). ``p`` is the model's own
``predict_proba``, so nothing is approximated about the model. The expectation
over orderings is estimated by sampling 32 permutations as 16 antithetic pairs
(each ordering and its reverse; Strumbelj & Kononenko 2014), the same ones for
every applicant (seeded), so an explanation is deterministic. The estimate is
exactly additive by construction — for every background applicant and ordering
the switches telescope to ``p(x) - p(r)`` — so

    base_value + sum_j phi_j == p(x),   base_value = mean_{r in R} p(r).

Who the players are
-------------------
``SHAPExplainer.over_inputs`` (what the API uses) attributes
over the applicant's INPUTS, with the features derived inside the value
function exactly as ``/score`` derives them, so every reason names something
the applicant supplied, never an engineered number such as
``monthly_payment_burden``. Two groups of fields
the API couples switch together as one player, so no recombination breaks their
rule: income with its cash flows and balance (inflow 0.78-1.22 x income / 12,
outflow a share of inflow, balance a multiple of income / 12, net cash flow their
difference) and the delinquency count with its recency (-1 exactly when the count
is 0).

The remaining joint rule (monthly payment at most 2.8 x monthly income) spans loan
amount, term and income, which stay separate players because a reason code that
merged them would say nothing useful. Recombining any training values cannot
break it (training maximum 2.67), so for an applicant inside the training range
no evaluated state is one the API would refuse. An applicant in the domain's 5%
margin (a $49,000 loan, say) can produce such a state when combined with a
low-income background applicant; the model reads it as its largest training
burden's straight-line extension, and the response's ``warnings`` already name
the out-of-range input. ``tests/test_shap_explainer.py`` checks both sides.

Accuracy (measured 2026-11-04 on the API model, 18 players): against shap's
KernelExplainer on the same function, background and players (8,192 samples, no
L1 regularisation), on 12 applicants including the 3 nearest the baseline and 3
not approved, every player worth more than 1 point agrees in sign and the
largest per-player gap is 0.44 point. The tests re-check with a 1.5-point
tolerance (the two estimators' Monte-Carlo error, with margin; 32 permutations
are within 0.46 point of a 256-permutation run). Cost: 32 x 1,000 x 19 = 608,000 model evaluations per explanation, in batches of
8 orderings (``/explain`` latency: README).

Exact additivity alone proves nothing about faithfulness: rescaling log-odds
TreeSHAP by (change in served probability) / (change in margin) is exactly additive
too, and that ratio explodes near the baseline. Hence the independent estimator above.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from creditlens.decision import DECISION_REASONS, probability_to_score, score_to_decision

logger = logging.getLogger(__name__)


@dataclass
class IndividualExplanation:
    """SHAP explanation for a single credit decision."""

    applicant_id: int | str
    predicted_score: int           # 0-1000
    predicted_default_prob: float  # 0-1
    decision: str                  # 'Approved' | 'Denied' | 'Review'
    base_value: float              # served P(default) at the expected margin (module doc)
    shap_values: dict[str, float]  # feature → contribution to served P(default);
                                   # base_value + sum == the unrounded P(default)
    top_positive_factors: list[dict]  # factors lowering default risk
    top_negative_factors: list[dict]  # factors increasing default risk
    explanation_text: str          # human-readable explanation
    # Whether the SHAP attribution succeeded. "ok" => shap_values/factors are
    # real; "unavailable" => SHAP computation failed and the factor lists are
    # empty. The SCORE/decision above stay valid either way (they come from the
    # model, not from SHAP). Callers/UI must show "explanation unavailable"
    # rather than silently presenting empty/zeroed factors as if the applicant
    # simply had no driving factors.
    status: str = "ok"             # 'ok' | 'unavailable'
    status_reason: str | None = None  # set when status == 'unavailable'

    def __repr__(self) -> str:
        tag = "" if self.status == "ok" else f" | {self.status}"
        return (
            f"IndividualExplanation(id={self.applicant_id} | "
            f"score={self.predicted_score} | {self.decision}{tag})"
        )


@dataclass
class GlobalExplanation:
    """Aggregate SHAP analysis across the full dataset."""

    mean_absolute_shap: pd.Series       # feature → mean |SHAP|
    mean_shap: pd.Series                # feature → mean SHAP (direction)
    top_features: list[str]             # ranked by importance
    shap_summary: pd.DataFrame          # full SHAP value matrix


# Orderings evaluated per predict call: bounds memory (8 x 1,000 x 19 rows at a time).
_ORDERINGS_PER_BATCH = 8


def background_sample(served: np.ndarray, size: int, offset: float = 0.5) -> np.ndarray:
    """Indices of ``size`` rows: every k-th by served probability, starting ``offset``
    of the way into the first stride. Its mean probability is the population's within a
    tenth of a point, which is what makes the baseline mean something: 50 random rows
    read anywhere from 10.1% to 19.5% (2026-11-04)."""
    if len(served) <= size:
        return np.arange(len(served))
    order = np.argsort(served, kind="stable")
    stride = len(served) / size
    return np.sort(order[((np.arange(size) + offset) * stride).astype(int)])


class ServedShapleyExplainer:
    """Interventional Shapley values of a model's served P(default), estimated
    by antithetic permutation sampling against a fixed background (see the
    module docstring). Model-agnostic: it only calls ``predict_proba``.

    Exposes the duck type ``SHAPExplainer`` uses: ``expected_value`` and
    ``shap_values(X)``.
    """

    def __init__(
        self,
        predict_default: Any,
        background: np.ndarray,
        n_permutations: int = 32,
        seed: int = 0,
        groups: list[list[int]] | None = None,
    ) -> None:
        if n_permutations < 2 or n_permutations % 2:
            raise ValueError("n_permutations must be a positive even number (antithetic pairs)")
        self._f = predict_default
        self.background = np.asarray(background, dtype=float)
        self.n_permutations = n_permutations
        self.seed = seed
        n_columns = self.background.shape[1]
        # A player is a set of columns that switch together; by default each
        # column is its own player.
        self.groups = groups if groups is not None else [[j] for j in range(n_columns)]
        if sorted(c for cols in self.groups for c in cols) != list(range(n_columns)):
            raise ValueError("every column must belong to exactly one player")
        owner = np.empty(n_columns, dtype=int)
        for g, cols in enumerate(self.groups):
            owner[cols] = g
        self._owner = owner
        # In float64: a float32 model output averaged in float32 drifts ~1e-8
        # from the float64 telescoping sum, breaking exact additivity.
        self.expected_value = float(np.mean(np.asarray(self._f(self.background), dtype=float)))

    def _masks(self, n_features: int, n_permutations: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        perms = []
        for _ in range(n_permutations // 2):
            order = rng.permutation(n_features)
            perms += [order, order[::-1]]
        perms = np.array(perms)
        # masks[a, k, j] = 1 when feature j is among the first k of ordering a.
        rank = np.argsort(perms, axis=1)
        masks = (rank[:, None, :] < np.arange(n_features + 1)[None, :, None]).astype(float)
        return perms, masks

    def shap_values(self, X: np.ndarray, n_permutations: int | None = None) -> np.ndarray:
        """One row per applicant, one column per player."""
        X = np.asarray(X, dtype=float)
        n_perm = n_permutations or self.n_permutations
        n_players, n_columns = len(self.groups), X.shape[1]
        perms, masks = self._masks(n_players, n_perm)
        columns = masks[:, :, self._owner]  # each column follows its player
        bg = self.background
        out = np.zeros((len(X), n_players), dtype=float)
        for i, x in enumerate(X):
            phi = np.zeros(n_players)
            for start in range(0, n_perm, _ORDERINGS_PER_BATCH):
                part = columns[start:start + _ORDERINGS_PER_BATCH]
                # (orderings, background, steps 0..M, columns): background values,
                # switched to x's one player at a time in each ordering.
                states = bg[None, :, None, :] * (1.0 - part[:, None]) + x * part[:, None]
                v = np.asarray(self._f(states.reshape(-1, n_columns)), dtype=float)
                v = v.reshape(len(part), len(bg), n_players + 1)
                step = np.diff(v, axis=2).mean(axis=1)  # (orderings, k): gain at step k
                for a in range(len(part)):
                    phi[perms[start + a]] += step[a]
            out[i] = phi / n_perm
        return out


class SHAPExplainer:
    """
    SHAP explainability wrapper for CreditLens models.

    Parameters
    ----------
    model : LogisticScorer | XGBoostScorer | NeuralNetScorer
        Trained credit scoring model.
    feature_names : list[str]
        Feature column names, one player each (``X`` is then the model's
        feature matrix). ``SHAPExplainer.over_inputs`` is how the API explains:
        ``X`` is raw applicant rows and the players are the applicant's inputs.

    Decisions use the one shared policy in ``creditlens.decision`` (the same
    cutoffs the API applies), so an explanation can never state a different
    decision than ``/score`` returns for the same applicant.

    Examples
    --------
    >>> explainer = SHAPExplainer(xgb_model, feature_names)
    >>> explainer.fit(X_train)
    >>> explanation = explainer.explain_individual(X_test.iloc[[0]], applicant_id=42)
    >>> print(explanation.explanation_text)
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        engineer: Any = None,
        players: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.model = model
        # Without an engineer: one player per model feature, and X is the
        # model's feature matrix. With one (``over_inputs``): the players are
        # the applicant's input fields (some grouped) and X is raw applicant
        # rows; the features are derived inside the value function.
        self.engineer = engineer
        self.players = dict(players) if players else {f: (f,) for f in feature_names}
        self.feature_names = list(self.players)
        self.input_fields = [f for fields in self.players.values() for f in fields]
        self._explainer = None
        self._shap_values_cache: np.ndarray | None = None
        # Which attribution fit() built; "" until then. See the module docstring.
        self.method: str = ""

    @classmethod
    def over_inputs(cls, model: Any, engineer: Any) -> "SHAPExplainer":
        """Attribute over the applicant's input fields (``attribution_players``
        in features/engineering.py), the way the API explains a decision."""
        from creditlens.features.engineering import attribution_players

        return cls(model, engineer.feature_names, engineer=engineer, players=attribution_players())

    METHOD = ("interventional Shapley values of the served P(default): "
              "32 antithetic permutations x up to 1,000 fixed background applicants "
              "(every k-th by served probability)")
    INPUT_METHOD = ("interventional Shapley values of the served P(default) over the "
                    "applicant's inputs (income with its cash flows and balance, and the "
                    "delinquency count with its recency, as one player each; net cash flow derived): "
                    "32 antithetic permutations x up to 1,000 fixed background applicants "
                    "(every k-th by served probability)")
    BACKGROUND_SIZE = 1000
    GLOBAL_MAX_ROWS = 200
    GLOBAL_PERMUTATIONS = 8
    GLOBAL_BACKGROUND = 100

    def fit(self, X_background: pd.DataFrame) -> "SHAPExplainer":
        """
        Fix the background the attributions are relative to: 1,000 rows of
        ``X_background``, every k-th by served probability (all of them if it has
        fewer). Works for any model with ``predict_proba``: it is explained through
        the probability it serves.

        Parameters
        ----------
        X_background : pd.DataFrame
            Training rows (the API passes all of them).

        Returns
        -------
        self
        """
        def predict_default(A: np.ndarray) -> np.ndarray:
            return self._served(pd.DataFrame(A, columns=self.input_fields))

        rows = self._columns(X_background)
        rows = rows[background_sample(predict_default(rows), self.BACKGROUND_SIZE)]

        groups = [[self.input_fields.index(f) for f in fields] for fields in self.players.values()]
        self._explainer = ServedShapleyExplainer(predict_default, rows, groups=groups)
        self.method = self.METHOD if self.engineer is None else self.INPUT_METHOD
        return self

    def _columns(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[self.input_fields].to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def _served(self, X: pd.DataFrame) -> np.ndarray:
        """The P(default) /score serves for these rows (raw rows when the
        players are inputs: the features are derived exactly as /score does)."""
        if self.engineer is not None:
            X = self.engineer.transform(X)[0]
        return self.model.predict_proba(X)[:, 1]

    def explain_individual(
        self,
        X: pd.DataFrame,
        applicant_id: int | str = 0,
    ) -> IndividualExplanation:
        """
        Generate SHAP explanation for a single applicant.

        Parameters
        ----------
        X : pd.DataFrame
            Single-row DataFrame for the applicant.
        applicant_id : int | str
            Identifier for the applicant.

        Returns
        -------
        IndividualExplanation
        """
        if self._explainer is None:
            raise RuntimeError("Call fit() before explain_individual().")

        proba = float(self._served(X)[0])
        score = probability_to_score(proba)
        decision = score_to_decision(score)

        # Compute SHAP values. On failure we must NOT silently substitute zeros
        # and present them as "no driving factors" — that is indistinguishable
        # from a genuine all-flat explanation and is a wrong-but-quiet output on
        # a lending surface. Instead surface an explicit degraded explanation:
        # the score/decision (from the model, not SHAP) still stand, but the
        # factor breakdown is marked unavailable with a reason.
        try:
            shap_vals = np.asarray(self._explainer.shap_values(self._columns(X)))[0]
            base_val = float(self._explainer.expected_value)
        except Exception as e:
            logger.error(
                "SHAP computation failed for applicant %s: %s. Returning a "
                "degraded explanation (status='unavailable'); score/decision "
                "are still valid.", applicant_id, e,
            )
            unavailable_text = (
                f"Credit Decision: {decision}\n"
                f"Credit Score: {score}/1000\n"
                f"Estimated Default Probability: {proba:.1%}\n\n"
                "Feature-level explanation is temporarily unavailable "
                "(SHAP attribution failed). The score and decision above remain "
                "valid; the factor breakdown could not be computed for this "
                "request."
            )
            return IndividualExplanation(
                applicant_id=applicant_id,
                predicted_score=score,
                predicted_default_prob=round(proba, 4),
                decision=decision,
                base_value=float(proba),
                shap_values={},
                top_positive_factors=[],
                top_negative_factors=[],
                explanation_text=unavailable_text,
                status="unavailable",
                status_reason=f"SHAP computation failed: {e}",
            )

        # Build feature → SHAP mapping. Cast to native Python float: numpy
        # scalars are not JSON-encodable by FastAPI/Pydantic.
        feat_shap = {k: float(v) for k, v in zip(self.feature_names, shap_vals)}
        base_val = float(base_val)

        # Sort by absolute SHAP value
        sorted_feats = sorted(feat_shap.items(), key=lambda x: abs(x[1]), reverse=True)

        # Positive factors = features that DECREASE default prob (negative SHAP = lower risk)
        positive = [
            {"feature": k, "shap_value": round(v, 4), "impact": "reduces default risk"}
            for k, v in sorted_feats if v < -0.001
        ][:5]

        # Negative factors = features that INCREASE default prob
        negative = [
            {"feature": k, "shap_value": round(v, 4), "impact": "increases default risk"}
            for k, v in sorted_feats if v > 0.001
        ][:5]

        explanation_text = self._generate_explanation_text(
            score, proba, decision, base_val, positive, negative
        )

        return IndividualExplanation(
            applicant_id=applicant_id,
            predicted_score=score,
            predicted_default_prob=round(proba, 4),
            decision=decision,
            base_value=base_val,
            shap_values=feat_shap,
            top_positive_factors=positive,
            top_negative_factors=negative,
            explanation_text=explanation_text,
        )

    def explain_global(self, X: pd.DataFrame) -> GlobalExplanation:
        """
        Compute aggregate SHAP explanation over a dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset to explain (e.g., test set).

        Returns
        -------
        GlobalExplanation
        """
        if self._explainer is None:
            raise RuntimeError("Call fit() before explain_global().")

        try:
            # Global importances average over rows, which averages the
            # Monte-Carlo noise too: 200 rows x 8 permutations x 100 background
            # rows keeps the cost near 5 single explanations.
            rows = self._columns(X)[:min(self.GLOBAL_MAX_ROWS, len(X))]
            inner = self._explainer
            if len(inner.background) > self.GLOBAL_BACKGROUND:
                stride = len(inner.background) // self.GLOBAL_BACKGROUND
                inner = ServedShapleyExplainer(inner._f, inner.background[::stride], groups=inner.groups)
            sv = inner.shap_values(rows, n_permutations=self.GLOBAL_PERMUTATIONS)
        except Exception as e:
            # Do NOT silently zero: an all-zero matrix yields a meaningless
            # feature-importance ranking (every feature "equally unimportant")
            # that would be reported as if it were real. Fail loudly instead —
            # a corrupted global explanation must not pass unnoticed.
            logger.error("Global SHAP computation failed: %s", e)
            raise RuntimeError(f"Global SHAP computation failed: {e}") from e

        shap_df = pd.DataFrame(sv[:, :len(self.feature_names)], columns=self.feature_names)
        mean_abs = shap_df.abs().mean().sort_values(ascending=False)
        mean_shap = shap_df.mean().sort_values(ascending=False)

        return GlobalExplanation(
            mean_absolute_shap=mean_abs,
            mean_shap=mean_shap,
            top_features=mean_abs.index.tolist()[:10],
            shap_summary=shap_df,
        )

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _generate_explanation_text(
        self,
        score: int,
        proba: float,
        decision: str,
        base_value: float,
        positive: list[dict],
        negative: list[dict],
    ) -> str:
        """Human-readable explanation built ONLY from this applicant's score and
        SHAP attribution — nothing canned.

        Contributions are in percentage points of P(default) ("pts"), the space
        the attribution is exactly additive in: baseline + every contribution =
        the served probability.
        """
        def label(factor: dict) -> str:
            return factor["feature"].replace("_", " ").title()

        lines = [
            f"Credit Decision: {decision}",
            f"Credit Score: {score}/1000",
            f"Estimated Default Probability: {proba:.1%}",
            "",
            f"Baseline default probability {base_value:.1%}; this applicant's "
            f"features move it to {proba:.1%}.",
        ]
        if negative:
            lines.append("Largest factors raising the estimated risk:")
            lines += [f"  ✗ {label(f)} ({f['shap_value'] * 100:+.1f} pts)" for f in negative[:3]]
        if positive:
            lines.append("Largest factors lowering the estimated risk:")
            lines += [f"  ✓ {label(f)} ({f['shap_value'] * 100:+.1f} pts)" for f in positive[:3]]

        lines += ["", DECISION_REASONS[decision]]
        if decision == "Review":
            lines.append("This reference implementation has no review workflow.")
        elif decision == "Denied":
            lines.append(
                "The factors above are this model's largest contributions to the "
                "estimate (SHAP), not general credit advice."
            )
        return "\n".join(lines)
