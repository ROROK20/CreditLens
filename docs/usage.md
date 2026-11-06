# CreditLens Usage Guide

Concrete, copy-paste examples for using CreditLens as a Python library and over
its REST API. All examples run on locally generated synthetic data — no external
services or secrets are required.

## Installation

```bash
git clone https://github.com/ROROK20/CreditLens.git
cd CreditLens
pip install -e ".[xai,dev]"   # xai adds SHAP; dev adds pytest/httpx
```

## Python API

### 1. End-to-end: data → features → model → evaluation

```python
from creditlens.data.generator import (
    generate_credit_dataset, DatasetConfig, train_test_split,
)
from creditlens.features.engineering import FeatureEngineer
from creditlens.models.scorer import LogisticScorer, evaluate_model

# Generate reproducible synthetic data. (The README Quick Start runs this same
# pipeline and shows its exact, tested output.)
config = DatasetConfig(n_samples=10_000, seed=42)
df = generate_credit_dataset(config)
train_df, test_df = train_test_split(df, test_size=0.20, seed=42)

# Engineer features (traditional + alternative + derived)
engineer = FeatureEngineer(scale=True)
X_train, y_train = engineer.fit_transform(train_df)
X_test, y_test = engineer.transform(test_df)

# Train and evaluate the API's scorer: sign-constrained logistic regression
model = LogisticScorer().fit(X_train, y_train)

evaluation = evaluate_model(model, X_test, y_test, "LogisticScorer")
print(evaluation)
print("AUC-ROC:", round(evaluation.auc_roc, 3))
print("Gini:   ", round(evaluation.gini_coefficient, 3))
print("KS:     ", round(evaluation.ks_statistic, 3))
```

### 2. Explain a single decision with SHAP

```python
from creditlens.explainability.shap_explainer import SHAPExplainer

# Attribute over the applicant's inputs, the way the API does: raw rows go in,
# and features are derived inside the value function. Income moves with its cash
# flows and balance, and the delinquency count with its recency, as one factor each.
explainer = SHAPExplainer.over_inputs(model, engineer)
explainer.fit(train_df)   # 1,000 of these become the background

explanation = explainer.explain_individual(test_df.iloc[[0]], applicant_id=1)

print(explanation)                  # IndividualExplanation(id=1 | score=... | <decision>)
print("Decision:", explanation.decision)
print("Score:   ", explanation.predicted_score, "/ 1000")
print("P(default):", round(explanation.predicted_default_prob, 3))
print(explanation.explanation_text)

# Ranked factors
for factor in explanation.top_negative_factors[:3]:
    print("↑ risk:", factor)
for factor in explanation.top_positive_factors[:3]:
    print("↓ risk:", factor)
```

### 3. Global feature importance

```python
global_exp = explainer.explain_global(test_df.iloc[:1000])   # uses the first 200 rows
print("Top features:", global_exp.top_features[:5])
print(global_exp.mean_absolute_shap.head())
```

### 4. Run a fairness audit

```python
from creditlens.fairness.metrics import FairnessAuditor

auditor = FairnessAuditor()   # audits the rule /score applies: score >= 600
y_proba = model.predict_proba(X_test)[:, 1]

report = auditor.audit(
    y_test.values,
    y_proba,
    test_df["demographic_group"].values,
)

print(report)          # FairnessReport(PASS | FAIL | INCONCLUSIVE | ...)
for name, check in report.checks.items():
    print(f"{name:19s} {check.value:.3f}  95% CI {check.ci_low:.3f}-{check.ci_high:.3f}  {check.verdict}")
for group, m in report.group_metrics.items():
    print(group, "n =", m.n_samples, "defaulted =", m.n_positive,
          "P(approved | repaid) =", round(m.approval_rate_repaid, 3))
for rec in report.recommendations:
    print("•", rec)
```

Each criterion is judged on its confidence interval, not its point estimate:
**PASS** when every group pair's interval lies inside the threshold, **FAIL**
when one lies entirely beyond it, **INCONCLUSIVE** otherwise. Equal
opportunity is the standard lending form, the gap in P(approved | repaid).
Equalized odds' interval adds the bounds of its two signed gaps, so it is conservative
and its lower bound reaches 0 when both signed intervals span 0.
`FairnessAuditor(threshold=0.20)` audits a *hypothetical* cutoff
P(default) < 0.20 instead of the served rule. The README Quick Start shows
real, dated output for this exact sequence.

## REST API

Start the server. It trains a model on synthetic data at startup (10,000 rows,
seed 42), so every endpoint below works immediately:

```bash
uvicorn creditlens.api.app:app --reload --port 8002
# Interactive OpenAPI docs: http://localhost:8002/docs
```

### Health check

```bash
curl http://localhost:8002/health
# {"status": "ok", "model_loaded": true, "service": "CreditLens", "version": "0.1.0"}
```

### Retrain the model (optional, off by default)

You do not need this to use the API. `POST /train` is **off by default** and answers `403`: it is unauthenticated
and replaces the model every later request is scored with. Start the server
with `CREDITLENS_ENABLE_TRAIN=1` to enable it (startup training does not need
it). It takes a JSON **body** (query-string parameters are ignored). With no
body it rebuilds exactly the startup model, so the scores documented below
still reproduce after it:

```bash
curl -X POST http://localhost:8002/train \
  -H "Content-Type: application/json" \
  -d '{"n_samples": 10000, "seed": 42}'
# {"status": "trained", "n_train": 8000, "n_test": 2000, "auc_roc": 0.8564,
#  "fairness_pass": false, "fairness_verdict": "INCONCLUSIVE"}
```

`n_samples` must be 2,000–50,000 (below 2,000 the 20% audit set is too small
to say anything about a group). Any other value or seed retrains a different model and
the scores below will differ from the documented ones.

### Score a single applicant

```bash
curl -X POST http://localhost:8002/score \
  -H "Content-Type: application/json" \
  -d '{"annual_income": 75000, "loan_amount": 12000, "loan_term_months": 36,
       "fico_score": 680,
       "debt_to_income": 0.32, "employment_years": 4, "credit_utilization": 0.45,
       "credit_age_months": 84, "num_hard_inquiries": 2,
       "num_delinquencies": 0, "months_since_delinquency": -1,
       "monthly_inflow": 6250, "monthly_outflow": 4800, "avg_account_balance": 8500,
       "paycheck_regularity": 0.8, "bill_payment_consistency": 0.8,
       "nsf_count": 0, "overdraft_days": 0, "rent_payment_score": 0.8,
       "utility_payment_score": 0.8, "banking_tenure_months": 12,
       "savings_rate": 0.1}'
# {"credit_score": 951, "default_probability": 0.0491,
#  "decision": "Approved", "decision_reason": "Score at or above the approval cutoff (600).",
#  "warnings": []}
```

Reproduced from code 2026-11-04 (`tests/test_docs.py` re-runs both this and
the batch example below). Every input the model uses is required; a missing
one is a `422`, never a silent default, and nothing is imputed. The library agrees:
`FeatureEngineer.transform` raises `ValueError` on an absent or NaN input unless you
opt in with `impute_missing=True`, which warns what it filled. Net cash flow
is not an input: the engine always derives it as `monthly_inflow -
monthly_outflow`, and sending it is a `422`. Types are strict: send numbers as JSON numbers and
integer fields as JSON integers. There is no interest-rate
input (a `422` if sent): the offered rate is the lender's own risk-based
price, so the model does not use it. `demographic_group` is accepted and
ignored. Bounds: see `ApplicantFeatures` in `creditlens/api/app.py` and the
tables in [SECURITY.md](../SECURITY.md).

### Score a batch

```bash
curl -X POST http://localhost:8002/score/batch \
  -H "Content-Type: application/json" \
  -d '{"applicants": [
        {"annual_income": 75000, "loan_amount": 12000, "loan_term_months": 36,
         "fico_score": 680,
         "debt_to_income": 0.32, "employment_years": 4, "credit_utilization": 0.45,
         "credit_age_months": 84, "num_hard_inquiries": 2,
         "num_delinquencies": 0, "months_since_delinquency": -1,
         "monthly_inflow": 6250, "monthly_outflow": 4800, "avg_account_balance": 8500,
         "paycheck_regularity": 0.8, "bill_payment_consistency": 0.8,
         "nsf_count": 0, "overdraft_days": 0, "rent_payment_score": 0.8,
         "utility_payment_score": 0.8, "banking_tenure_months": 12,
         "savings_rate": 0.1}
      ]}'
# {"count": 1, "results": [{"index": 0, "credit_score": 951, ...}]}
```

### Explain a decision

```bash
curl -X POST http://localhost:8002/explain/1 \
  -H "Content-Type: application/json" \
  -d '{"annual_income": 30000, "loan_amount": 25000, "loan_term_months": 36,
       "fico_score": 560,
       "debt_to_income": 0.62, "employment_years": 1, "credit_utilization": 0.92,
       "credit_age_months": 18, "num_hard_inquiries": 6,
       "num_delinquencies": 2, "months_since_delinquency": 5,
       "monthly_inflow": 2500, "monthly_outflow": 2400, "avg_account_balance": 1000,
       "paycheck_regularity": 0.5, "bill_payment_consistency": 0.5,
       "nsf_count": 3, "overdraft_days": 8, "rent_payment_score": 0.5,
       "utility_payment_score": 0.5, "banking_tenure_months": 6,
       "savings_rate": 0.0}'
# {"applicant_id": 1, "credit_score": 0, "decision": "Denied", "warnings": [...],
#  "explanation_text": "...", "top_negative_factors": [...], "top_positive_factors": [...],
#  "base_value": 0.1492, "shap_values": {...}, "attribution_players": {...}, ...}
```

Each key of `shap_values` is one of the applicant's inputs, or a group of inputs
that move together: `attribution_players` lists them (`income_cash_flow_and_balance`
is `annual_income`, `monthly_inflow`, `monthly_outflow` and `avg_account_balance`;
`delinquency_history` is
`num_delinquencies` and `months_since_delinquency`). `base_value` plus the sum of
`shap_values` is the served P(default). `warnings` names the in-domain values few
training applicants share.

### Model metrics & fairness report

```bash
curl http://localhost:8002/model/metrics
# {"model_name": "LogisticScorer", "auc_roc": 0.8564, "auc_pr": 0.5713,
#  "gini_coefficient": 0.7127, "brier_score": 0.0917, "ks_statistic": 0.5542}

curl http://localhost:8002/fairness/report
# {"overall_verdict": "INCONCLUSIVE", "overall_pass": false,
#  "demographic_parity_diff": 0.0182, "equal_opportunity_diff": 0.0058,
#  "equalized_odds_diff": 0.0773, "disparate_impact_ratio": 0.9798,
#  "passes_disparate_impact": true}
```

The fairness report above is abridged (`tests/test_docs.py` checks every key
shown). The full report adds, per criterion, its `definition`, `value`,
simultaneous 95% `ci`, `threshold`, `verdict` and per-pair breakdown, and per
group `n_samples` / `n_repaid` / `n_defaulted`, approval rates overall, among
repaid and among defaulted, and calibration (observed minus mean predicted
default rate, with its interval). On this 2,000-applicant audit set
demographic parity (PASS, CI 0–0.0778), equal opportunity (PASS, CI 0–0.0487)
and disparate impact (PASS, CI 0.9149–1) are decided, and equalized odds straddles
its threshold, 0.0773 with CI 0–0.2953. Under `population` the report repeats the audit on
400,000 fresh applicants: equalized odds 0.1103 (CI 0.0911–0.1292), and group C defaulting at
21.3% against 14.5% predicted. At 2,000 audited the straddle is mostly sample size, and
separately the injected gap sits near the gate on that large draw (README "What the demo
reports, and why"). Reproduced 2026-11-04.

## Web Demo

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3003
```

The demo offers an interactive applicant form, a score rail marking the 450 and 600 cutoffs, a
SHAP-style contribution waterfall, and a fairness dashboard with three preset
profiles (Strong / Median Applicant / Weak).

> ⚠️ CreditLens uses synthetic data and is a research/educational reference
> implementation. Do not use it to make real lending decisions.
