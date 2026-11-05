"""POST /train safety.

- /train is unauthenticated and replaces the model every later request is
  scored with, so it is OFF unless the operator opts in with
  CREDITLENS_ENABLE_TRAIN=1 (tests/conftest.py opts in for the suite).
- The minimum is 2,000 rows (a 400-applicant audit set): far fewer would audit
  fairness on a handful of group-C applicants.
- The model, its explainer and its reports are replaced together, so a failed SHAP
  setup cannot leave an explainer of another model next to the new one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from creditlens.api import app as api

APPLICANT = {
    "annual_income": 75000, "loan_amount": 12000, "loan_term_months": 36, "fico_score": 680, "debt_to_income": 0.32,
    "employment_years": 4, "credit_utilization": 0.45, "credit_age_months": 84, "num_hard_inquiries": 2, "num_delinquencies": 0,
    "months_since_delinquency": -1, "monthly_inflow": 6250, "monthly_outflow": 4800,
    "avg_account_balance": 8500, "paycheck_regularity": 0.8, "bill_payment_consistency": 0.8,
    "nsf_count": 0, "overdraft_days": 0, "rent_payment_score": 0.8,
    "utility_payment_score": 0.8, "banking_tenure_months": 12, "savings_rate": 0.1,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


@pytest.mark.parametrize("value", [None, "", "0", "false", "no"])
def test_train_is_off_unless_the_operator_opts_in(client, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CREDITLENS_ENABLE_TRAIN", raising=False)
    else:
        monkeypatch.setenv("CREDITLENS_ENABLE_TRAIN", value)
    before = client.post("/score", json=APPLICANT).json()
    res = client.post("/train", json={"n_samples": 2000, "seed": 7})
    assert res.status_code == 403
    assert "CREDITLENS_ENABLE_TRAIN" in res.json()["detail"]
    # The served model was not touched.
    assert client.post("/score", json=APPLICANT).json() == before


def test_train_works_when_enabled(client, monkeypatch):
    monkeypatch.setenv("CREDITLENS_ENABLE_TRAIN", "1")
    res = client.post("/train")
    assert res.status_code == 200
    assert res.json()["n_train"] + res.json()["n_test"] == api.DEFAULT_TRAIN_SAMPLES


@pytest.mark.parametrize("n", [100, 1999])
def test_train_refuses_a_dataset_too_small_to_audit(client, monkeypatch, n):
    monkeypatch.setenv("CREDITLENS_ENABLE_TRAIN", "1")
    assert client.post("/train", json={"n_samples": n}).status_code == 422


def test_failed_explainer_setup_never_leaves_the_old_explainer_behind(client, monkeypatch):
    from creditlens.explainability.shap_explainer import SHAPExplainer

    api._startup_train()
    old_explainer = api._model_state["explainer"]

    def broken_fit(self, X_background):
        raise RuntimeError("simulated explainer setup failure")

    monkeypatch.setattr(SHAPExplainer, "fit", broken_fit)
    api.train_model(n_samples=2000, seed=7)
    try:
        assert api._model_state["explainer"] is None
        assert api._model_state["explainer"] is not old_explainer
        res = client.post("/explain/0", json=APPLICANT)
        assert res.status_code == 503  # never the previous model's explanation
        assert client.post("/score", json=APPLICANT).status_code == 200  # the new model serves
    finally:
        monkeypatch.undo()
        api._startup_train()
    assert api._model_state["explainer"] is not None
