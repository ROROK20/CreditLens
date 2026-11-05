"""One decision policy (creditlens.decision), applied identically everywhere.

The fairness audit must approve exactly where the API approves, so /fairness/report
describes the decision /score returns.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from creditlens.api.app import DEFAULT_SEED, DEFAULT_TRAIN_SAMPLES, ApplicantFeatures, _startup_train, app
from creditlens.data.generator import DatasetConfig, generate_credit_dataset, train_test_split
from creditlens.decision import (
    APPROVE_MIN_SCORE, REVIEW_MIN_SCORE, probability_to_score, score_to_decision,
)

ROOT = Path(__file__).resolve().parents[1]


def test_decision_bands_at_their_boundaries():
    assert score_to_decision(APPROVE_MIN_SCORE) == "Approved"
    assert score_to_decision(APPROVE_MIN_SCORE - 1) == "Review"
    assert score_to_decision(REVIEW_MIN_SCORE) == "Review"
    assert score_to_decision(REVIEW_MIN_SCORE - 1) == "Denied"


def test_probability_to_score_scalar_and_batch_agree():
    probs = [0.0, 0.0777, 0.3995, 0.4, 0.4005, 0.5505, 1.0]
    batch = probability_to_score(probs)
    assert [probability_to_score(p) for p in probs] == batch.tolist()
    assert isinstance(probability_to_score(0.25), int)
    # 0.3995 -> 600.5 -> 600 and 0.5505 -> 449.5 -> 450 (half to even); 0.4005 is
    # 599.4999... in float64 -> 599.
    assert batch.tolist() == [1000, 922, 600, 600, 599, 450, 0]


def test_frontend_mirrors_the_python_cutoffs():
    """frontend/lib/scoring.ts carries its own copy (TypeScript cannot import
    Python); this fails the moment the two drift."""
    ts = (ROOT / "frontend" / "lib" / "scoring.ts").read_text(encoding="utf-8")
    approve = re.search(r"export const APPROVE_CUTOFF = (\d+);", ts)
    review = re.search(r"export const REVIEW_CUTOFF = (\d+);", ts)
    assert approve and review
    assert int(approve.group(1)) == APPROVE_MIN_SCORE
    assert int(review.group(1)) == REVIEW_MIN_SCORE


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_fairness_report_audits_the_decisions_the_api_returns(client):
    """Send the audit's own test set through /score/batch: each group's
    approval_rate in /fairness/report must be the share of that group the API
    actually returned as "Approved"."""
    _startup_train()  # order-independent: earlier tests may have retrained
    df = generate_credit_dataset(DatasetConfig(n_samples=DEFAULT_TRAIN_SAMPLES, seed=DEFAULT_SEED))
    _, test_df = train_test_split(df, test_size=0.20, seed=DEFAULT_SEED)
    # Native Python scalars: JSON round-trips float64 exactly via repr.
    applicants = [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        # Only the API's fields: the generator also records the lender's rate,
        # open accounts, subscriptions and loan purpose, none of which is an
        # input (no effect on the synthetic label).
        for row in test_df[[c for c in test_df.columns if c in ApplicantFeatures.model_fields]].to_dict(orient="records")
    ]
    # /score/batch takes at most 1,000 applicants; the audit set is 2,000.
    decisions = []
    for start in range(0, len(applicants), 1000):
        res = client.post("/score/batch", json={"applicants": applicants[start:start + 1000]})
        assert res.status_code == 200, res.text[:300]
        decisions += [r["decision"] for r in res.json()["results"]]

    report = client.get("/fairness/report").json()["group_metrics"]
    groups = test_df["demographic_group"].tolist()
    assert set(report) == set(groups)
    for g, metrics in report.items():
        mine = [d for d, gg in zip(decisions, groups) if gg == g]
        served_rate = sum(d == "Approved" for d in mine) / len(mine)
        assert metrics["n_samples"] == len(mine)
        assert metrics["approval_rate"] == pytest.approx(served_rate, abs=5e-5), g


def test_decision_reasons_state_the_rule_that_decided():
    """Every reason is true for every applicant it is attached to: it states the
    band rule."""
    from creditlens.decision import DECISION_REASONS

    assert DECISION_REASONS == {
        "Approved": "Score at or above the approval cutoff (600).",
        "Review": "Score in the manual-review band (450-599).",
        "Denied": "Score below the review cutoff (450).",
    }
    assert f"({APPROVE_MIN_SCORE})" in DECISION_REASONS["Approved"]
    assert f"({REVIEW_MIN_SCORE}-{APPROVE_MIN_SCORE - 1})" in DECISION_REASONS["Review"]
    assert f"({REVIEW_MIN_SCORE})" in DECISION_REASONS["Denied"]


def test_frontend_mirrors_the_decision_reasons():
    from creditlens.decision import DECISION_REASONS

    ts = (ROOT / "frontend" / "lib" / "scoring.ts").read_text(encoding="utf-8")
    block = ts[ts.index("export const DECISION_REASONS"):]
    block = block[:block.index("};")]
    mirrored = dict(re.findall(r'(\w+): "([^"]+)"', block))
    assert mirrored == DECISION_REASONS


def test_api_decision_reason_is_the_band_rule(client):
    weak = {"annual_income": 30000, "loan_amount": 25000, "debt_to_income": 0.62,
            "employment_years": 1, "fico_score": 560, "credit_utilization": 0.92, "credit_age_months": 18, "monthly_inflow": 2500,
            "monthly_outflow": 2400, "avg_account_balance": 1000, "num_hard_inquiries": 6,
            "loan_term_months": 36, "num_delinquencies": 0, "months_since_delinquency": -1,
            "paycheck_regularity": 0.8, "bill_payment_consistency": 0.8, "nsf_count": 0,
            "overdraft_days": 0, "rent_payment_score": 0.8, "utility_payment_score": 0.8,
            "banking_tenure_months": 12, "savings_rate": 0.1,}
    _startup_train()
    res = client.post("/score", json=weak).json()
    assert res["decision"] == "Denied"
    assert res["decision_reason"] == "Score below the review cutoff (450)."
