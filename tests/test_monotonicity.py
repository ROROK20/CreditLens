"""Monotonicity of the served credit score in features with an unambiguous risk
direction (creditlens.features.engineering.RISK_DIRECTION).

An unconstrained model can learn noise a credit reviewer finds at once, such as a
HIGHER credit utilization or overdraft count raising the score. These tests sweep
inputs and require the served probability / score to move only one way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from creditlens.api.app import app
from creditlens.data.generator import DatasetConfig, generate_credit_dataset
from creditlens.features.engineering import INPUT_RISK_DIRECTION, RISK_DIRECTION, FeatureEngineer
from creditlens.models.scorer import LogisticScorer, XGBoostScorer


@pytest.fixture(scope="module", params=["logistic", "xgboost"])
def fitted(request):
    df = generate_credit_dataset(DatasetConfig(n_samples=1500, seed=5))
    engineer = FeatureEngineer(scale=True)
    X, y = engineer.fit_transform(df)
    # XGBoost default: calibrate=True, monotone=True
    model = LogisticScorer() if request.param == "logistic" else XGBoostScorer(n_estimators=60)
    model.fit(X, y)
    return model, X


def test_every_direction_names_a_real_feature():
    df = generate_credit_dataset(DatasetConfig(n_samples=200, seed=1))
    engineer = FeatureEngineer()
    engineer.fit_transform(df)
    assert set(RISK_DIRECTION) <= set(engineer.feature_names)
    assert set(RISK_DIRECTION.values()) <= {-1, 1}


def test_served_probability_is_monotone_in_every_constrained_feature(fitted):
    """Sweep each constrained (scaled) feature over its training range for 25
    applicants, everything else fixed: the served, calibrated, fold-averaged
    P(default) may move only in the declared direction."""
    model, X = fitted
    rows = X.iloc[:25]
    violations = []
    for feature, direction in RISK_DIRECTION.items():
        grid = np.linspace(X[feature].min(), X[feature].max(), 15)
        sweep = pd.concat([rows.assign(**{feature: v}) for v in grid])
        p = model.predict_proba(sweep)[:, 1].reshape(len(grid), len(rows))
        steps = np.diff(p, axis=0) * direction  # >= 0 everywhere if monotone
        # 1e-12: at P near 1 a float rounds by an ulp either way.
        if (steps < -1e-12).any():
            violations.append(f"{feature}: worst step {steps.min():.4f}")
    assert not violations, violations


# --- end to end, through the API, in raw units ---------------------------------

README_APPLICANT = {
    "annual_income": 75000, "loan_amount": 12000, "fico_score": 680,
    "debt_to_income": 0.32, "employment_years": 4, "credit_utilization": 0.45,
    "credit_age_months": 84, "monthly_inflow": 6250, "monthly_outflow": 4800,
    "avg_account_balance": 8500, "num_hard_inquiries": 2,
    # Every model input is required, so these are supplied explicitly.
    "loan_term_months": 36, "num_delinquencies": 0, "months_since_delinquency": -1,
    "paycheck_regularity": 0.8, "bill_payment_consistency": 0.8, "nsf_count": 0,
    "overdraft_days": 0, "rent_payment_score": 0.8, "utility_payment_score": 0.8,
    "banking_tenure_months": 12, "savings_rate": 0.1,
}
DELINQUENT = {**README_APPLICANT, "num_delinquencies": 2, "months_since_delinquency": 12}

# raw field -> (base applicant, grid). Every raw input is monotone end to end, and
# every grid lies inside the validity domain (data/domain.py): beyond it the API
# returns 422 rather than extrapolating.
RAW_SWEEPS = {
    "annual_income": (README_APPLICANT, [15000, 35000, 50000, 75000, 120000, 250000]),
    "loan_amount": (README_APPLICANT, [1000, 5000, 12000, 20000, 40000]),
    "loan_term_months": (README_APPLICANT, [12, 24, 36, 48, 60]),
    "fico_score": (README_APPLICANT, [350, 450, 560, 620, 680, 720, 780, 850]),
    "debt_to_income": (README_APPLICANT, [0.01, 0.1, 0.2, 0.32, 0.45, 0.6, 0.74]),
    "credit_utilization": (README_APPLICANT, [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.6, 0.8, 0.98]),
    "num_delinquencies": (DELINQUENT, [1, 2, 3, 4, 5]),
    "months_since_delinquency": (DELINQUENT, [0, 3, 6, 12, 24, 60, 120]),
    "num_hard_inquiries": (README_APPLICANT, [0, 1, 2, 4, 7, 8]),
    "credit_age_months": (README_APPLICANT, [3, 12, 36, 84, 180, 360]),
    "employment_years": (README_APPLICANT, [0, 0.5, 2, 4, 10, 20]),
    "monthly_inflow": (README_APPLICANT, [4900, 5200, 6250, 7000, 7600]),
    "monthly_outflow": (README_APPLICANT, [2950, 3500, 4000, 4800, 6000]),
    "paycheck_regularity": (README_APPLICANT, [0.0, 0.3, 0.6, 0.8, 0.95, 1.0]),
    "bill_payment_consistency": (README_APPLICANT, [0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]),
    "nsf_count": (README_APPLICANT, [0, 1, 2, 3, 5, 6]),
    "overdraft_days": (README_APPLICANT, [0, 2, 5, 8, 10]),
    "rent_payment_score": (README_APPLICANT, [0.13, 0.4, 0.7, 0.9, 1.0]),
    "utility_payment_score": (README_APPLICANT, [0.06, 0.4, 0.7, 0.9, 1.0]),
    "banking_tenure_months": (README_APPLICANT, [1, 6, 12, 48, 120, 240]),
    "avg_account_balance": (README_APPLICANT, [2400, 5000, 8500, 12000, 19000]),
    "savings_rate": (README_APPLICANT, [0.0, 0.02, 0.1, 0.25, 0.38]),
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_raw_sweeps_cover_every_constrained_raw_input():
    # Net cash flow is derived, not an input; the inflow and outflow
    # sweeps move it, and the engineered-level test above sweeps it directly.
    raw_inputs = set(INPUT_RISK_DIRECTION)
    assert raw_inputs == set(RAW_SWEEPS)


@pytest.mark.parametrize("field", sorted(RAW_SWEEPS))
def test_served_score_is_monotone_in_raw_input(client, field):
    base, grid = RAW_SWEEPS[field]
    scores = []
    for value in grid:
        payload = {**base, field: value}
        if field == "annual_income":
            # Inflow and the balance are income-linked (data/domain.py), so a
            # higher income arrives with proportionally higher flows and balance.
            scale = value / base["annual_income"]
            payload["monthly_inflow"] = base["monthly_inflow"] * scale
            payload["monthly_outflow"] = base["monthly_outflow"] * scale
            payload["avg_account_balance"] = base["avg_account_balance"] * scale
        res = client.post("/score", json=payload)
        assert res.status_code == 200, res.text
        scores.append(res.json()["credit_score"])
    score_direction = -INPUT_RISK_DIRECTION[field]  # score = 1000 * (1 - P(default))
    steps = [(b - a) * score_direction for a, b in zip(scores, scores[1:])]
    assert min(steps) >= 0, f"{field} grid {grid} -> scores {scores}"
