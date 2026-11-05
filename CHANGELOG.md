# Changelog

All notable changes to this project are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — first public release

Released 2026-11-04.

### What ships

- **Scoring API** (FastAPI): `POST /score`, `POST /score/batch`, `POST /explain/{id}`,
  `GET /fairness/report`, `GET /model/metrics`, `GET /health`. It trains a
  sign-constrained logistic regression on seeded synthetic data at startup, so every
  endpoint works with no manual step.
- **Explanations**: interventional Shapley values of the served P(default), over the
  applicant's own inputs. `base_value` plus the contributions equals the served
  probability, and the suite compares them with shap's KernelExplainer.
- **Fairness audit**: demographic parity, equal opportunity, equalized odds and disparate
  impact on the decision the API serves, each with a bootstrap interval and a
  PASS / FAIL / INCONCLUSIVE verdict, plus calibration by group and the same audit on a
  fresh draw of 400,000 applicants.
- **Synthetic data generator** with a published label function, and an opt-in
  `group_c_fico_shift` that plants a disparity the audit should catch.
- **Web demo** (Next.js 15): a scoring form, score gauge, contribution chart and fairness
  card. With no backend it runs the engine's recorded model in the browser and says so.
- **Library**: `LogisticScorer`, `XGBoostScorer` and an optional `NeuralNetScorer` behind
  one interface, `FeatureEngineer`, `SHAPExplainer` and `FairnessAuditor`.
- **Checks**: a pytest suite that re-runs the documented numbers, and `e2e/smoke.mjs`,
  which drives the real API and web UI in a real browser.

### What a user must know

- The data is synthetic and the project is a reference implementation. It is not a
  lending system and not financial advice; do not use it for real credit decisions.
- A FICO score is required, so it does not score the credit-invisible.
- Every model input is required. A missing or unknown field is a `422`, nothing is
  imputed, and the library's `FeatureEngineer.transform` refuses an absent or NaN input
  unless you pass `impute_missing=True`.
- Inputs are bounded to the training data's range plus a 5% margin, with five joint rules
  on income, cash flow, balance and loan payment. SECURITY.md has the exact table.
- The demo's own audit reads INCONCLUSIVE. At 2,000 audited applicants that is mostly
  sample size, and on the large draw equalized odds sits at about 0.11 against its 0.10
  gate. These are screening diagnostics, not a legal test.
- The approval cutoff (score 600) is illustrative, not a risk appetite. There is no
  pricing model and no APR.
- `POST /train` answers `403` unless the server is started with
  `CREDITLENS_ENABLE_TRAIN=1`: it is unauthenticated and replaces the model for every
  later caller.
- The documented numbers reproduce under the exact pins in `requirements.txt` on
  Python 3.12. The test suite fails if one stops reproducing.

### Known limits

- Stated cash flow can be gamed inside the rules. Every input is self-reported, so a
  lender must verify cash flow independently; the README measures the effect.
- `/explain` lists the largest contributions. It is not a compliant adverse-action notice.
- Contributions are associations in this model, not causes.
- The API keeps its model in one process, has no authentication or rate limiting, and is
  a single-worker demo. Harden it before exposing it publicly.
