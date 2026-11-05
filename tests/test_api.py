"""
Endpoint tests for the CreditLens FastAPI app.

Uses fastapi.testclient.TestClient with a context manager so the startup event
(which trains a small synthetic model) fires before the requests run. Asserts
each endpoint returns 200 with correctly-shaped, non-empty real-engine JSON.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import creditlens.api.app as api_module
from creditlens.api.app import app

SAMPLE_APPLICANT = {
    "annual_income": 75000,
    "loan_amount": 12000,
    "loan_term_months": 36,
    "debt_to_income": 0.32,
    "employment_years": 4,
    "fico_score": 720,
    "credit_utilization": 0.25,
    "num_delinquencies": 0,
    "months_since_delinquency": -1,
    "num_hard_inquiries": 1,
    "credit_age_months": 84,
    "monthly_inflow": 6250,
    "monthly_outflow": 4800,
    "paycheck_regularity": 0.92,
    "bill_payment_consistency": 0.88,
    "nsf_count": 0,
    "overdraft_days": 0,
    "rent_payment_score": 0.95,
    "utility_payment_score": 0.9,
    "banking_tenure_months": 48,
    "avg_account_balance": 8500,
    "savings_rate": 0.1,
    "demographic_group": "",
}


@pytest.fixture(scope="module")
def client():
    # Context manager triggers the @app.on_event("startup") training hook,
    # so the model is loaded before any request runs.
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["service"] == "CreditLens"
    # Model is trained on startup.
    assert body["model_loaded"] is True


def test_score_returns_real_prediction(client):
    res = client.post("/score", json=SAMPLE_APPLICANT)
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {
        "credit_score",
        "default_probability",
        "decision",
        "decision_reason",
        "warnings",
    }
    assert body["warnings"] == []  # every value inside the training data's central 99%
    assert 0 <= body["credit_score"] <= 1000
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["decision"] in {"Approved", "Review", "Denied"}
    assert isinstance(body["decision_reason"], str) and body["decision_reason"]


def test_explain_returns_shap_factors(client):
    res = client.post("/explain/0", json=SAMPLE_APPLICANT)
    assert res.status_code == 200
    body = res.json()
    assert body["applicant_id"] == 0
    assert 0 <= body["credit_score"] <= 1000
    assert isinstance(body["explanation_text"], str) and body["explanation_text"]
    assert isinstance(body["top_positive_factors"], list)
    assert isinstance(body["top_negative_factors"], list)
    # At least one side should carry real SHAP factors.
    factors = body["top_positive_factors"] + body["top_negative_factors"]
    assert factors, "expected at least one SHAP factor"
    for f in factors:
        assert "feature" in f and "shap_value" in f
    # The explanation status is surfaced; on the happy path it is "ok".
    assert body["explanation_status"] == "ok"
    assert body["explanation_status_reason"] is None


def test_fairness_report_shape(client):
    res = client.get("/fairness/report")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["overall_pass"], bool)
    for key in (
        "demographic_parity_diff",
        "equal_opportunity_diff",
        "disparate_impact_ratio",
    ):
        assert isinstance(body[key], (int, float))
    assert isinstance(body["group_metrics"], dict)
    assert body["group_metrics"], "expected per-group metrics"


def test_model_metrics(client):
    res = client.get("/model/metrics")
    assert res.status_code == 200
    body = res.json()
    assert body["model_name"]
    assert 0.0 <= body["auc_roc"] <= 1.0


def test_batch_score(client):
    res = client.post(
        "/score/batch", json={"applicants": [SAMPLE_APPLICANT, SAMPLE_APPLICANT]}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2


# A weak profile: low FICO, high utilization/DTI, delinquencies and NSF hits —
# scores far enough below SAMPLE_APPLICANT to land in a different decision band.
WEAK_APPLICANT = {
    **SAMPLE_APPLICANT,
    "fico_score": 540,
    "credit_utilization": 0.95,
    "debt_to_income": 0.62,
    "num_delinquencies": 4,
    "months_since_delinquency": 2,
    "num_hard_inquiries": 8,
    "nsf_count": 6,
    # 10 days: the validity domain's edge. The balance has to match income too.
    "overdraft_days": 10,
    "paycheck_regularity": 0.45,
    "bill_payment_consistency": 0.40,
    "rent_payment_score": 0.35,
    "utility_payment_score": 0.35,
    "annual_income": 28000,
    "monthly_inflow": 2300,
    "monthly_outflow": 2250,
    "avg_account_balance": 900,
    "savings_rate": 0.01,
    "employment_years": 0.5,
}


def test_batch_decisions_match_single_score_decisions(client):
    """/score/batch must use the SAME decision bands as /score: same scores, same
    decisions, index for index."""
    applicants = [SAMPLE_APPLICANT, WEAK_APPLICANT, SAMPLE_APPLICANT]

    singles = []
    for applicant in applicants:
        res = client.post("/score", json=applicant)
        assert res.status_code == 200
        singles.append(res.json())

    res = client.post("/score/batch", json={"applicants": applicants})
    assert res.status_code == 200
    batch = res.json()["results"]

    assert [r["index"] for r in batch] == [0, 1, 2]
    for single, row in zip(singles, batch):
        assert row["credit_score"] == single["credit_score"]
        assert row["decision"] == single["decision"]
        assert row["warnings"] == single["warnings"]

    # Non-vacuity: the two profiles must not both sit in the same band, or the
    # assertion above would hold under any ladder.
    assert singles[0]["decision"] != singles[1]["decision"], (
        f"expected different decision bands, got {singles[0]['decision']} "
        f"and {singles[1]['decision']} (scores "
        f"{singles[0]['credit_score']} / {singles[1]['credit_score']})"
    )


# --- Security: bound unauthenticated inputs so they cannot cause DoS ---


def test_train_rejects_huge_n_samples(client):
    """An unbounded n_samples would let an unauthenticated caller request a
    gigantic synthetic dataset + model fit (CPU/memory DoS). It must be
    rejected cleanly with 422, not hang or 500."""
    res = client.post("/train", json={"n_samples": 10_000_000})
    assert res.status_code == 422


def test_train_rejects_too_small_n_samples(client):
    res = client.post("/train", json={"n_samples": 1})
    assert res.status_code == 422


def test_train_accepts_in_range_n_samples(client):
    """The smallest legitimate value still trains. tests/test_train_endpoint.py
    pins the floor of 2,000."""
    res = client.post("/train", json={"n_samples": 2000})
    assert res.status_code == 200
    assert res.json()["status"] == "trained"


def test_batch_score_rejects_oversized_list(client):
    """An oversized applicants list must be rejected with 422 before any
    DataFrame / model work happens."""
    oversized = {"applicants": [SAMPLE_APPLICANT] * 1001}
    res = client.post("/score/batch", json=oversized)
    assert res.status_code == 422


def test_batch_score_allows_max_list(client):
    """Exactly at the cap is still allowed (legitimate behavior preserved)."""
    res = client.post("/score/batch", json={"applicants": [SAMPLE_APPLICANT] * 1000})
    assert res.status_code == 200
    assert res.json()["count"] == 1000


def test_bodyless_train_rebuilds_exactly_the_startup_model(client):
    """POST /train with no body must rebuild the SAME model the server trained
    at startup, so the documented scores reproduce after a retrain."""
    from creditlens.api.app import _startup_train

    _startup_train()  # order-independent: earlier tests retrain on other sizes
    before = [client.post("/score", json=a).json() for a in (SAMPLE_APPLICANT, WEAK_APPLICANT)]

    res = client.post("/train")
    assert res.status_code == 200
    # The documented demo numbers are derived from the startup training set.
    assert res.json()["n_train"] + res.json()["n_test"] == api_module.DEFAULT_TRAIN_SAMPLES

    after = [client.post("/score", json=a).json() for a in (SAMPLE_APPLICANT, WEAK_APPLICANT)]
    assert after == before


@pytest.mark.parametrize("applicant", [SAMPLE_APPLICANT, WEAK_APPLICANT], ids=["sample", "weak"])
def test_explain_attribution_reconstructs_the_served_probability(client, applicant):
    """/explain carries the full attribution in the same space as the score:
    base_value + sum(shap_values) == /score's default_probability (which is
    rounded to 4 dp, hence the tolerance), and both endpoints agree."""
    explained = client.post("/explain/0", json=applicant).json()
    scored = client.post("/score", json=applicant).json()
    assert explained["attribution_space"] == "probability_of_default"
    rebuilt = explained["base_value"] + sum(explained["shap_values"].values())
    assert rebuilt == pytest.approx(scored["default_probability"], abs=5e-5 + 1e-9)
    assert explained["credit_score"] == scored["credit_score"]
    assert explained["decision"] == scored["decision"]


CHECKS = {"demographic_parity", "equal_opportunity", "equalized_odds", "disparate_impact"}
VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE"}


def test_fairness_report_carries_definitions_intervals_and_sizes(client):
    """Every gated criterion ships its definition, a confidence interval and a
    three-state verdict; every group its sample sizes."""
    from creditlens.api.app import _startup_train

    _startup_train()
    fr = client.get("/fairness/report").json()
    assert fr["overall_verdict"] in VERDICTS
    assert fr["overall_pass"] == (fr["overall_verdict"] == "PASS")
    assert "score >= 600" in fr["decision_rule"]
    assert set(fr["checks"]) == CHECKS
    for name, check in fr["checks"].items():
        assert check["verdict"] in VERDICTS and check["definition"], name
        low, high = check["ci"]
        assert low <= check["value"] <= high, name
        assert fr["passes_" + name] == (check["verdict"] == "PASS")
    assert "P(approved | repaid" in fr["checks"]["equal_opportunity"]["definition"]
    assert sum(m["n_samples"] for m in fr["group_metrics"].values()) == api_module.DEFAULT_TRAIN_SAMPLES // 5
    for g, m in fr["group_metrics"].items():
        assert m["n_repaid"] + m["n_defaulted"] == m["n_samples"], g
        assert 0 <= m["approval_rate_repaid"] <= 1, g


def test_fairness_report_serializes_undefined_rates_as_null(client):
    """A group with no defaulters has no P(approved | defaulted). JSON has no
    NaN (Starlette refuses to encode one), so it must arrive as null — not as a
    500, and not as a fake 0."""
    import numpy as np

    from creditlens.api import app as api
    from creditlens.fairness.metrics import FairnessAuditor

    groups = np.array(["A"] * 100 + ["C"] * 100)
    y_true = np.array([0] * 80 + [1] * 20 + [0] * 100)
    y_proba = np.where(np.arange(200) % 7 == 0, 0.9, 0.1)
    saved = api._model_state.get("fairness_report")
    api._model_state["fairness_report"] = FairnessAuditor().audit(y_true, y_proba, groups)
    try:
        res = client.get("/fairness/report")
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["group_metrics"]["C"]["approval_rate_defaulted"] is None
        assert body["group_metrics"]["C"]["n_defaulted"] == 0
        assert body["checks"]["equalized_odds"]["value"] is None
        assert body["checks"]["equalized_odds"]["verdict"] == "INCONCLUSIVE"
    finally:
        api._model_state["fairness_report"] = saved
