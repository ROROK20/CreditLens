"""No model input may silently take a favourable value.

Every model input is required, so omitting bad news cannot improve a decision. Net cash
flow, the identity inflow - outflow, is not an input at all, and neither is the interest
rate, the lender's own risk-based price. Nothing is ever imputed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from creditlens.api.app import DEFAULT_SEED, DEFAULT_TRAIN_SAMPLES, ApplicantFeatures, _model_state, app
from creditlens.data.generator import DatasetConfig, generate_credit_dataset, train_test_split
from creditlens.features.engineering import MODEL_INPUTS, NUMERIC_FEATURES, RISK_DIRECTION

ROOT = Path(__file__).resolve().parents[1]

# Every derogatory field present and bad.
BAD = {
    "annual_income": 75000, "loan_amount": 12000, "loan_term_months": 36, "fico_score": 600, "debt_to_income": 0.45,
    "employment_years": 4, "credit_utilization": 0.7, "credit_age_months": 84, "num_hard_inquiries": 6, "num_delinquencies": 4,
    "months_since_delinquency": 3, "monthly_inflow": 6250, "monthly_outflow": 4800,
    "avg_account_balance": 8500, "paycheck_regularity": 0.4, "bill_payment_consistency": 0.4,
    # overdraft_days 10: the validity domain's edge (25 is outside the training
    # data, a 422).
    "nsf_count": 6, "overdraft_days": 10, "rent_payment_score": 0.8,
    "utility_payment_score": 0.8, "banking_tenure_months": 12, "savings_rate": 0.1,
}
OMITTED = ("num_delinquencies", "months_since_delinquency", "nsf_count", "overdraft_days",
                       "num_hard_inquiries", "bill_payment_consistency", "paycheck_regularity")
RAW_MODEL_INPUTS = sorted(MODEL_INPUTS)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_the_reviewers_applicant_cannot_improve_by_omission(client):
    full = client.post("/score", json=BAD).json()
    assert full["decision"] == "Denied"
    omitted = {k: v for k, v in BAD.items() if k not in OMITTED}
    res = client.post("/score", json=omitted)
    assert res.status_code == 422  # never a score built on what was left out
    missing = {e["loc"][-1] for e in res.json()["detail"] if e["type"] == "missing"}
    assert missing == set(OMITTED)
    assert client.post("/explain/2", json=omitted).status_code == 422


@pytest.mark.parametrize("field", RAW_MODEL_INPUTS)
def test_every_model_input_is_required(client, field):
    payload = {k: v for k, v in BAD.items() if k != field}
    res = client.post("/score", json=payload)
    assert res.status_code == 422, field
    assert [e["loc"][-1] for e in res.json()["detail"]] == [field]


def test_only_the_documented_inputs_are_optional():
    schema = ApplicantFeatures.model_json_schema()
    optional = set(schema["properties"]) - set(schema["required"])
    assert optional == {"demographic_group"}  # not a model input


def test_interest_rate_is_not_a_model_input():
    """The lender's own price is not evidence about the applicant."""
    assert "interest_rate" not in NUMERIC_FEATURES
    assert "interest_rate" not in RISK_DIRECTION
    _ = _model_state  # the startup model below must agree
    with TestClient(app) as c:
        assert "interest_rate" not in _model_state["engineer"].feature_names
        res = c.post("/score", json={**BAD, "interest_rate": 0.2})
        assert res.status_code == 422  # rejected, never silently ignored
        assert res.json()["detail"][0]["loc"][-1] == "interest_rate"


def test_responses_carry_no_imputation(client):
    """Nothing is filled in, so there is nothing to disclose."""
    assert set(client.post("/score", json=BAD).json()) == {
        "credit_score", "default_probability", "decision", "decision_reason", "warnings"}
    explained = client.post("/explain/0", json=BAD).json()
    assert "imputed_fields" not in explained
    assert "Not supplied" not in explained["explanation_text"]


def test_the_live_audit_audits_exactly_what_the_api_serves(client):
    """Every one of the 2,000 audited applicants, sent the way the web form sends
    an applicant (every model input, no rate), must score exactly what the
    audit computed for them. With the rate imputed on the form's path, 47
    decisions differed (all upward)."""
    from creditlens.api.app import _startup_train

    _startup_train()
    df = generate_credit_dataset(DatasetConfig(n_samples=DEFAULT_TRAIN_SAMPLES, seed=DEFAULT_SEED))
    _, test_df = train_test_split(df, test_size=0.20, seed=DEFAULT_SEED)
    X, _ = _model_state["engineer"].transform(test_df)
    audited = np.round((1 - _model_state["model"].predict_proba(X)[:, 1]) * 1000).astype(int)

    sent = test_df[[c for c in test_df.columns if c in ApplicantFeatures.model_fields
                    and c not in ("demographic_group", "monthly_net_cashflow")]]
    applicants = [{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
                  for row in sent.to_dict(orient="records")]
    served = []  # /score/batch takes at most 1,000 applicants; the audit set is 2,000
    for start in range(0, len(applicants), 1000):
        chunk = client.post("/score/batch", json={"applicants": applicants[start:start + 1000]}).json()
        served += [r["credit_score"] for r in chunk["results"]]
    np.testing.assert_array_equal(served, audited)


def test_the_form_sends_every_model_input_and_nothing_else():
    """The web form's payload is exactly the model's raw inputs: nothing the
    model uses is left to a default, and nothing it ignores is sent."""
    ts = (ROOT / "frontend" / "lib" / "api.ts").read_text(encoding="utf-8")
    body = ts[ts.index("function toApiApplicant"):]
    body = body[body.index("return {"):body.index("};")]
    sent = set(re.findall(r"^\s+(\w+): .+,$", body, re.M))
    assert sent == set(RAW_MODEL_INPUTS)


def test_frontend_sends_only_user_supplied_values():
    """The web demo hard-coded overdraft_days 0, utility_payment_score 0.85,
    savings_rate 0.1, num_subscriptions 3 and loan_purpose — values the user
    never gave, two of which the Weak preset's explanation then cited as
    risk-lowering reasons. Every field it sends must come from the form."""
    ts = (ROOT / "frontend" / "lib" / "api.ts").read_text(encoding="utf-8")
    body = ts[ts.index("function toApiApplicant"):]
    body = body[body.index("return {"):body.index("};")]
    sent = dict(re.findall(r"^\s+(\w+): (.+),$", body, re.M))
    required = set(ApplicantFeatures.model_json_schema()["required"])
    assert required <= set(sent), sorted(required - set(sent))
    for field, expr in sent.items():
        assert expr.startswith("a.") or "? a." in expr, f"{field} is not user-supplied: {expr}"
