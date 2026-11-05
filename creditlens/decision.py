"""
The credit decision policy — defined ONCE.

Every surface that turns a default probability into a score or a decision uses
this module: the REST API (``/score``, ``/score/batch``, ``/explain``), the SHAP
explainer's decision text, the scorers' ``predict_score()``, and the fairness
auditor — which must audit the rule the API actually applies.

The web demo mirrors the two cutoffs in ``frontend/lib/scoring.ts``
(``APPROVE_CUTOFF`` / ``REVIEW_CUTOFF``); ``tests/test_decision.py`` fails if
that copy drifts from this one.
"""

from __future__ import annotations

import numpy as np

# The cutoffs are ILLUSTRATIVE. APPROVE_MIN_SCORE = 600 means P(default) <= 0.40,
# about 2.7x the 14.9% training default rate, so the demo approves ~89% of
# applicants (1,775 of the 2,000 audited, 2026-11-04). There is no economic rationale
# behind it: a lender derives a cutoff from expected loss, pricing and capital. The
# point here is only that the API, the explanation and the fairness audit all
# apply the SAME rule.
SCORE_SCALE = 1000
APPROVE_MIN_SCORE = 600  # score >= 600            -> "Approved"
REVIEW_MIN_SCORE = 450   # 450 <= score < 600      -> "Review"; below -> "Denied"

DECISIONS = ("Approved", "Review", "Denied")

# Risk bands, as score cutoffs nested inside the decision bands. Each is defined by
# P(default) against the 14.9% base rate: at most 5% (a third of it), at most 15%
# (about it), at most 40% (the approval cutoff), then the review band and the denied
# band. score_band_table() measures each band's observed default rate on the audit
# set, and a test requires them to order.
RISK_BANDS = ((950, "Very low"), (850, "Low"), (APPROVE_MIN_SCORE, "Elevated"),
              (REVIEW_MIN_SCORE, "High"), (0, "Very high"))

# What a decision MEANS, stated as the rule that produced it — true for every
# applicant it is attached to. The per-applicant WHY is the SHAP attribution (/explain).
DECISION_REASONS = {
    "Approved": f"Score at or above the approval cutoff ({APPROVE_MIN_SCORE}).",
    "Review": f"Score in the manual-review band ({REVIEW_MIN_SCORE}-{APPROVE_MIN_SCORE - 1}).",
    "Denied": f"Score below the review cutoff ({REVIEW_MIN_SCORE}).",
}


def probability_to_score(p_default):
    """Credit score on the 0-1000 scale: ``round((1 - P(default)) * 1000)``.

    Higher = safer. Rounds half to even (numpy and Python ``round`` agree), so a
    scalar and the same value inside a batch always map to the same score.
    Returns an ``int`` for a scalar input and an int array otherwise.
    """
    scores = np.round((1.0 - np.asarray(p_default, dtype=float)) * SCORE_SCALE).astype(int)
    return int(scores) if scores.ndim == 0 else scores


def score_to_decision(score: int) -> str:
    """Map a 0-1000 score to "Approved" / "Review" / "Denied"."""
    if score >= APPROVE_MIN_SCORE:
        return "Approved"
    if score >= REVIEW_MIN_SCORE:
        return "Review"
    return "Denied"


def is_approved(scores) -> np.ndarray:
    """Boolean mask of the scores the policy approves (``score >= 600``).

    "Review" is not an approval: the fairness audit's favourable outcome is the
    decision the API returns as "Approved".
    """
    return np.asarray(scores) >= APPROVE_MIN_SCORE


def score_band_table(scores, defaulted) -> list[dict]:
    """Each risk band on an audit set: its score range, size and observed default rate."""
    scores, defaulted = np.asarray(scores), np.asarray(defaulted)
    rows, upper = [], SCORE_SCALE
    for lower, label in RISK_BANDS:
        in_band = (scores >= lower) & (scores <= upper)
        n = int(in_band.sum())
        rows.append({"label": label, "min_score": lower, "max_score": upper, "n": n,
                     "share": round(n / len(scores), 4),
                     "observed_default_rate": round(float(defaulted[in_band].mean()), 4) if n else None})
        upper = lower - 1
    return rows
