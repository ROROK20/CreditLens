"""Record the live engine's model and answers for the no-backend page.

    python frontend/scripts/record_engine.py      # or: make snapshot

Writes frontend/lib/engineSnapshot.json. The served model is a logistic regression,
so its weights let a browser evaluate it exactly; tests/test_engine_snapshot.py fails
if this file drifts from the engine.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from fastapi.testclient import TestClient

from creditlens.api import app as api
from creditlens.data import domain
from creditlens.data.generator import DatasetConfig, generate_credit_dataset, train_test_split

OUT = Path(__file__).resolve().parents[1] / "lib" / "engineSnapshot.json"

DEFAULT = {
    "annual_income": 75000, "loan_amount": 12000, "loan_term_months": 36, "debt_to_income": 0.32,
    "employment_years": 4, "fico_score": 720, "credit_utilization": 0.25, "num_delinquencies": 0,
    "months_since_delinquency": -1, "num_hard_inquiries": 1, "credit_age_months": 84,
    "monthly_inflow": 6250, "monthly_outflow": 4800, "paycheck_regularity": 0.92,
    "bill_payment_consistency": 0.88, "nsf_count": 0, "overdraft_days": 0, "rent_payment_score": 0.95,
    "utility_payment_score": 0.85, "banking_tenure_months": 48, "avg_account_balance": 8500, "savings_rate": 0.1,
}
STRONG = {**DEFAULT, "fico_score": 800, "credit_utilization": 0.1, "debt_to_income": 0.2,
          "paycheck_regularity": 0.98, "bill_payment_consistency": 0.96, "utility_payment_score": 0.97,
          "savings_rate": 0.2, "annual_income": 120000, "monthly_inflow": 10000, "monthly_outflow": 7000,
          "employment_years": 10, "avg_account_balance": 25000}
MEDIAN = {**DEFAULT, "fico_score": 677, "credit_utilization": 0.31, "debt_to_income": 0.26,
          "paycheck_regularity": 0.68, "bill_payment_consistency": 0.74, "overdraft_days": 2,
          "rent_payment_score": 0.77, "utility_payment_score": 0.73, "savings_rate": 0.11,
          "annual_income": 59000, "monthly_inflow": 4900, "monthly_outflow": 3500, "employment_years": 3.5,
          "avg_account_balance": 8000, "loan_amount": 10000, "credit_age_months": 36,
          "banking_tenure_months": 24}
WEAK = {**DEFAULT, "fico_score": 580, "credit_utilization": 0.75, "debt_to_income": 0.55, "num_delinquencies": 3,
        "months_since_delinquency": 8, "paycheck_regularity": 0.6, "bill_payment_consistency": 0.55,
        "nsf_count": 4, "overdraft_days": 10, "utility_payment_score": 0.5, "rent_payment_score": 0.6,
        "savings_rate": 0.01, "annual_income": 35000, "monthly_inflow": 2900, "monthly_outflow": 2700,
        "employment_years": 1, "avg_account_balance": 1200}
REFUSED = {
    "inflow does not match income": {**DEFAULT, "monthly_inflow": 12000, "monthly_outflow": 9000,
                                     "avg_account_balance": 8500},
    "spending beyond inflow": {**DEFAULT, "monthly_outflow": 7400},
    "almost no spending": {**DEFAULT, "monthly_outflow": 410},
    "balance far above income": {**DEFAULT, "avg_account_balance": 170000},
    "balance far below income": {**DEFAULT, "avg_account_balance": 500},
    "payment beyond income": {**DEFAULT, "annual_income": 12000, "monthly_inflow": 1000, "monthly_outflow": 800,
                              "avg_account_balance": 1500, "loan_amount": 49000, "loan_term_months": 9},
    "net cash flow beyond its range": {**DEFAULT, "annual_income": 600000, "monthly_inflow": 61000,
                                       "monthly_outflow": 30000, "avg_account_balance": 60000},
    # Inside every joint rule; each breaks one field's own bound.
    "balance under its floor at low income": {**DEFAULT, "annual_income": 15000, "monthly_inflow": 1250,
                                              "monthly_outflow": 960, "avg_account_balance": 465},
    "FICO under its floor": {**DEFAULT, "fico_score": 349},
    "delinquencies over their ceiling": {**DEFAULT, "num_delinquencies": 6, "months_since_delinquency": 8},
    "recency without a delinquency": {**DEFAULT, "months_since_delinquency": 8},
    "a delinquency without a recency": {**DEFAULT, "num_delinquencies": 1},
    "a fractional count": {**DEFAULT, "nsf_count": 1.5},
}
# Scored, each with warnings the page has to show.
WARNED = {
    "outside the central 99%": {**DEFAULT, "fico_score": 400, "nsf_count": 5},
    "inflow past the training data": {**DEFAULT, "monthly_inflow": 7625, "monthly_outflow": 5800},
    "tight net cash flow": {**DEFAULT, "monthly_inflow": 1250, "monthly_outflow": 1170, "annual_income": 15000,
                            "avg_account_balance": 1700},
    "heavy payment": {**DEFAULT, "annual_income": 15000, "monthly_inflow": 1250, "monthly_outflow": 960,
                      "avg_account_balance": 1700, "loan_amount": 24000, "loan_term_months": 12},
    "cash flow at its favourable edge": {**WEAK, "monthly_inflow": 3500, "monthly_outflow": 1800,
                                         "avg_account_balance": 8700},
}


def _group(m: dict) -> dict:
    return {"group": None, "nSamples": m["n_samples"], "nDefaulted": m["n_defaulted"],
            "approvalRate": m["approval_rate"], "approvalRateRepaid": m["approval_rate_repaid"],
            "defaultRate": m["default_rate"], "avgScore": m["avg_score"],
            "meanPredicted": m["mean_predicted_default"], "calibrationGap": m["calibration_in_the_large"],
            "calibrationLow": m["calibration_ci"][0], "calibrationHigh": m["calibration_ci"][1]}


def _check(c: dict) -> dict:
    return {"value": c["value"], "ciLow": c["ci"][0], "ciHigh": c["ci"][1], "verdict": c["verdict"]}


def fairness_snapshot(report: dict) -> dict:
    population = report["population"]
    worst_group, worst = max(population["group_metrics"].items(),
                             key=lambda kv: abs(kv[1]["calibration_in_the_large"]))
    return {
        "verdict": report["overall_verdict"],
        "checks": {name: _check(c) for name, c in report["checks"].items()},
        "groupMetrics": [{**_group(m), "group": g} for g, m in report["group_metrics"].items()],
        "population": {
            "nSamples": population["n_samples"], "verdict": population["overall_verdict"],
            "equalizedOdds": _check(population["checks"]["equalized_odds"]),
            "mostMispredicted": {
                "group": worst_group, "defaultRate": worst["default_rate"],
                "meanPredicted": worst["mean_predicted_default"], "gap": worst["calibration_in_the_large"],
                "ciLow": worst["calibration_ci"][0], "ciHigh": worst["calibration_ci"][1]},
        },
    }


def model_snapshot() -> dict:
    """The weights per raw unit of each model feature, for the training scale."""
    model, engineer = api._model_state["model"], api._model_state["engineer"]
    scale = dict(zip(engineer.feature_names, engineer._scaler.scale_))
    mean = dict(zip(engineer.feature_names, engineer._scaler.mean_))
    weights = {name: float(w) / float(scale[name]) for name, w in model.coefficients().items()}
    intercept = model.intercept() - sum(float(w) * float(mean[n]) / float(scale[n])
                                        for n, w in model.coefficients().items())
    return {"intercept": float(intercept), "weights": weights,
            "feature_means": {n: float(v) for n, v in mean.items()}}


def rules_snapshot() -> dict:
    return {"inflow_to_monthly_income": list(domain.INFLOW_TO_MONTHLY_INCOME),
            "outflow_to_inflow": list(domain.OUTFLOW_TO_INFLOW),
            "balance_to_monthly_income": list(domain.BALANCE_TO_MONTHLY_INCOME),
            "net_cashflow": list(domain.VALIDITY_DOMAIN["monthly_net_cashflow"]),
            "max_payment_burden": domain.MAX_PAYMENT_BURDEN,
            "fields": {name: list(bounds) for name, bounds in domain.VALIDITY_DOMAIN.items()
                       if name != "monthly_net_cashflow"},
            "integer_fields": list(domain.INTEGER_FIELDS),
            "warning_band": {name: list(band) for name, band in domain.WARNING_BAND.items()},
            "inflow_warning": list(domain.INFLOW_TO_MONTHLY_INCOME_WARNING),
            "outflow_warning": list(domain.OUTFLOW_TO_INFLOW_WARNING),
            "balance_warning": list(domain.BALANCE_TO_MONTHLY_INCOME_WARNING),
            "payment_burden_warning": domain.PAYMENT_BURDEN_WARNING,
            "cash_flow_edge": domain.CASH_FLOW_EDGE}


def audit_applicants(n: int = 6) -> list[dict]:
    df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=api.DEFAULT_SEED))
    _, test = train_test_split(df, test_size=0.20, seed=api.DEFAULT_SEED)
    out = []
    for _, row in test.iloc[:n].iterrows():
        out.append({k: (int(row[k]) if api.ApplicantFeatures.model_fields[k].annotation is int else float(row[k]))
                    for k in api.ApplicantFeatures.model_fields if k != "demographic_group"})
    return out


def record() -> dict:
    with TestClient(api.app) as client:
        api._startup_train()
        vectors = []
        named = [("default", DEFAULT), ("strong", STRONG), ("median", MEDIAN), ("weak", WEAK)]
        named += [(f"audit {i}", a) for i, a in enumerate(audit_applicants())]
        named += list(WARNED.items())
        for name, applicant in named:
            body = client.post("/score", json=applicant).json()
            vectors.append({"name": name, "status": 200, "applicant": applicant,
                            "credit_score": body["credit_score"],
                            "default_probability": body["default_probability"],
                            "decision": body["decision"], "warnings": body["warnings"]})
        for name, applicant in REFUSED.items():
            vectors.append({"name": name, "status": client.post("/score", json=applicant).status_code,
                            "applicant": applicant})
        report = client.get("/fairness/report").json()
    return {"recorded": datetime.date.today().isoformat(), "seed": api.DEFAULT_SEED,
            "n_samples": api.DEFAULT_TRAIN_SAMPLES, "model": model_snapshot(), "rules": rules_snapshot(),
            "vectors": vectors, "score_bands": report["score_bands"], "fairness": fairness_snapshot(report)}


if __name__ == "__main__":
    OUT.write_text(json.dumps(record(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
