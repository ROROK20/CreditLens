# CreditLens Architecture

CreditLens is an explainable credit-scoring reference. It scores loan applicants
from traditional **and** alternative (cash-flow / behavioral) data, explains every
decision with SHAP, and audits model fairness across demographic groups. This
document describes the components and how data flows through them.

## High-Level Data Flow

```
                         ┌─────────────────────────┐
                         │  Synthetic Data Generator│
                         │  creditlens.data         │
                         │  (seeded; opt-in bias)   │
                         └────────────┬────────────┘
                                      │ DataFrame (27 raw features + group + target)
                                      ▼
                         ┌─────────────────────────┐
                         │   Feature Engineering    │
                         │  creditlens.features     │
                         │  traditional + alt +     │
                         │  derived, optional scale │
                         └────────────┬────────────┘
                            X, y       │
                                      ▼
                         ┌─────────────────────────┐
                         │      Credit Scorer       │
                         │  creditlens.models       │
                         │  LogisticScorer (served) │
                         │  XGBoost / NeuralNet     │
                         └───┬─────────┬─────────┬──┘
              predict_proba  │         │         │
            ┌────────────────┘         │         └───────────────┐
            ▼                          ▼                         ▼
 ┌────────────────────┐   ┌────────────────────────┐  ┌────────────────────┐
 │  SHAP Explainer    │   │   Fairness Auditor     │  │  Model Evaluation  │
 │ explainability/    │   │   fairness/metrics     │  │  evaluate_model()  │
 │ Individual+Global  │   │ parity / EO / DI       │  │  AUC, Gini, KS...  │
 └─────────┬──────────┘   └───────────┬────────────┘  └─────────┬──────────┘
           │                          │                         │
           └──────────────┬───────────┴─────────────────────────┘
                          ▼
              ┌─────────────────────────┐        ┌─────────────────────┐
              │      REST API           │◀──────▶│   Next.js Frontend  │
              │  creditlens.api (FastAPI)│  HTTP  │   frontend/ (demo)  │
              └─────────────────────────┘        └─────────────────────┘
```

## Components

### `creditlens.data` — Synthetic Data Generator
`generate_credit_dataset(config: DatasetConfig)` produces a stated ground truth, not
a simulation of lending: 27 raw features (income, FICO, utilization, delinquencies, cash
flow, bill-payment behavior, NSF/overdraft history, rent/utility payments,
banking tenure, etc.; 22 of them are model inputs, which the model reads as 19
features, the terms of the label), a demographic
group, and a binary default target. The features are mostly independent draws from
hand-chosen distributions: inflow, outflow and balance follow income, the offered rate
follows FICO, and delinquency recency follows the delinquency count. On the default
10,000 rows (2026-11-04) FICO's correlation is -0.007 with delinquencies, 0.009 with utilization
and -0.007 with income, and banking tenure exceeds credit age on 37.7% of rows. The group is
drawn independently of every feature. `DatasetConfig` controls `n_samples`, `seed`, and two group-C
settings. The default label is `default_log_odds()`: every model input has a
documented, modest effect on it (`LABEL_EFFECTS`), and the model's monotone
constraints follow those directions. Four generated fields have no effect and
are not model inputs: open accounts, subscriptions, loan purpose and the
offered interest rate (README "Model behaviour"). The two group-C settings:

- `group_c_bias` (default 0.08, on): group C defaults more often at the same
  features. No feature carries it and the model never sees the group, so approval
  rates stay level; it shows up as group-C miscalibration and as a gap in approval
  among defaulters (README "What the demo reports, and why").
- `group_c_fico_shift` (default 0, off): understates group C's *reported* FICO
  by this many points while default risk follows the true score, a
  measurement bias that makes the model deny creditworthy group-C applicants,
  which the fairness audit flags (README Quick Start: FAIL at 80 points). `0`
  leaves the dataset byte-identical.

`train_test_split` provides a seeded, reproducible split.

### `creditlens.features` — Feature Engineering
`FeatureEngineer` turns raw applicant data into a model-ready feature matrix:

- **Traditional** signals: FICO, debt-to-income, utilization, delinquencies.
- **Alternative** signals: cash flow, bill-payment consistency, NSF count,
  overdrafts, rent and utility payments, savings, balance, banking tenure.
- **Derived** signals: the new loan's payment burden (loan amount, term and
  income reach the model only through it) and net cash flow (inflow and outflow
  likewise). Delinquency recency is capped at 24 months, where its effect on the
  label ends.

The 19 features are exactly the terms of the synthetic label, each once, so an
input has one path into the model.

`fit_transform(df)` learns column structure and optional scaling on the training
set; `transform(df)` applies the same transformation to new data, and refuses
(`ValueError`, naming each) any required input that is absent or NaN: scoring without
one asserts something about the applicant. `FeatureEngineer(impute_missing=True)` fills
them with training medians and warns what it filled, for analysis on incomplete data.
The learned columns are exposed via the `feature_names` property, which feeds the
explainer.

### `creditlens.models` — Scorers & Evaluation
- `LogisticScorer` — what the API serves: logistic regression fitted with scipy's
  L-BFGS-B over standardised inputs, each weight bounded to its direction in
  `features.engineering.RISK_DIRECTION`, deterministic. The label is a logistic
  model of these features, plus a group-C shift and noise the model never sees, so this
  is the correct form in the features it has. README "Why
  logistic regression" publishes the comparison with depth-1 trees and how far one
  small step can move a score; "Model behaviour" publishes how far each input's
  response tracks its true effect, and names those that miss.
- `XGBoostScorer` — gradient-boosted trees, monotone-constrained in the same
  directions, kept in the library and tested. README "Why logistic regression"
  compares a depth-1 configuration with the served model. The constructor defaults
  to depth 5 with isotonic calibration on 3 folds, for library use.
- `NeuralNetScorer` — an optional MLP with the same interface, for comparison.
  The API does not use it, and no ensembling code exists.
- `evaluate_model(model, X_test, y_test)` returns a `ModelEvaluation` with
  AUC-ROC, AUC-PR, Brier score, KS statistic, Gini coefficient, a classification
  report, and a confusion matrix.

Scores are reported on a 0–1000 scale derived from `(1 - default_probability)`.

### `creditlens.explainability` — SHAP Explainer
`SHAPExplainer.over_inputs(model, engineer)` wraps a trained model and the
engineer that derives its features, and attributes over the applicant's INPUTS,
which is how the API explains. (`SHAPExplainer(model, feature_names)` attributes
over the features of a matrix instead, for library use.)

- `fit(X_background)` fixes the background the attributions are relative to:
  1,000 of the rows passed in, every k-th by served probability (the API passes
  all its training rows), so the baseline is the population's mean.
- `explain_individual(X_row, applicant_id)` returns an `IndividualExplanation`
  with the predicted score, default probability, decision, base value, per-input
  SHAP contributions, ranked positive/negative factors, and human-readable text.
- **What the attributions are.** Interventional Shapley values of the SERVED
  P(default), the model's own `predict_proba`, relative to the fixed background:
  each factor's value is the average change in the served probability when it
  switches from a background applicant's value to this applicant's, over
  orderings of the factors. The features are derived inside the value function
  exactly as `/score` derives them. `ServedShapleyExplainer` estimates the average
  over orderings from 32 antithetic permutations (seeded, so an explanation is
  deterministic) in batches of 8 orderings, 608,000 rows in all: `/explain`'s
  median was 484 ms in-process (fastest 246 ms) on one contended machine (2026-11-04). The switches telescope, so `base_value + sum(contributions)` equals
  the served P(default) exactly, where `base_value` is the mean served P(default)
  over the background (0.149 for the API model, 2026-11-04).
- **Every reason is something the applicant supplied.** Attribution is over 18
  factors: 16 inputs alone, plus two groups the API couples, which switch together
  so no state breaks their rule: `income_cash_flow_and_balance` (inflow 0.78 to
  1.22 x income / 12, outflow 0.47 to 0.98 x inflow, balance 0.37 to 3.13 x income /
  12, net cash flow their difference) and
  `delinquency_history` (recency is -1 exactly when the count is 0). The payment
  rule (monthly payment at most 2.8 x monthly income) spans loan amount, term
  and income, kept separate because a merged reason would say nothing useful.
  Recombining training values cannot break it (training maximum 2.67); an applicant
  in the domain's 5% margin can produce such a state, which the logistic model scores
  on the same straight line, past where it has data. `tests/test_shap_explainer.py` puts real evaluated states
  through the API's validators.
- **Faithfulness is tested, not assumed.** Exact additivity does not make an
  attribution faithful: rescaling log-odds TreeSHAP by (change in served probability)
  / (change in margin) is exactly additive too, and that ratio explodes near the baseline.
  `tests/test_shap_explainer.py` compares the attribution with shap's
  KernelExplainer on the same function and factors, both over a 50-row slice of
  the API's background to keep the reference tractable (8,192 / 4,096
  samples, no L1 regularisation): on the API model (2026-11-04) every factor above
  1 point agrees in sign and the largest per-factor gap on 12 applicants (the 3
  nearest the baseline, 3 not approved, and others) is 0.44 point; the tests allow
  1.5 points, the two estimators' measured Monte-Carlo error with margin. Away from
  the baseline (|P - base| ≥ 5 points) the attribution mass is at most 10.7x the
  move (p99 7.8). Near the baseline the ratio is unbounded for any faithful
  attribution (cancelling factors over a move close to zero), so no all-rows bound
  is claimed.
- `POST /explain` returns `attribution_method`, `attribution_players`,
  `base_value` and the full `shap_values` so a client can check the sum itself. Every attributed input
  was supplied by the caller: the API imputes nothing.
- `explain_global(X)` returns a `GlobalExplanation` with mean |contribution|
  feature importances over at most 200 rows, using 8 permutations each (the
  mean over rows averages the Monte-Carlo noise).

Decision thresholds (defined once in `creditlens/decision.py`, shared by the
API, the explainer and the fairness audit): score ≥ 600 → **Approved**,
450–599 → **Review**, < 450 → **Denied**.

### `creditlens.fairness` — Fairness Auditor
`FairnessAuditor.audit(y_true, y_proba, groups)` audits the decision the API
returns (approved = score ≥ 600) on the full audit set and returns a
`FairnessReport` with four gated criteria, each the worst group pair's value:

- **Demographic parity** — gap in P(approved).
- **Equal opportunity** — gap in P(approved | repaid), the standard lending
  form (Hardt, Price & Srebro 2016).
- **Equalized odds** — gap in P(approved | repaid) plus gap in
  P(approved | defaulted).
- **Disparate impact** — min/max approval-rate ratio, screened at 0.80.

Each criterion (`report.checks`) carries its definition, point value, a
stratified-bootstrap interval simultaneous over group pairs (95%; equalized odds adds
the bounds of its two signed gaps, which is conservative), and a
three-state verdict: PASS / FAIL only when the interval clears the threshold,
otherwise INCONCLUSIVE — and always INCONCLUSIVE, with a `reason`, when there
are fewer than two groups or a group is too small to judge (under 30
applicants, or under 10 of a conditioning outcome). Per-group `GroupMetrics` carry the sample sizes
(applicants, repaid, defaulted, approved) and calibration within the group
(reported, not gated). Undefined rates (a group with no defaulters) are NaN in
Python and `null` on the wire, never 0.

### `creditlens.api` — FastAPI Service
`creditlens.api.app:app` exposes the engine over HTTP. The model state is held
in memory and trained on synthetic data at startup (or lazily on the first scoring
request if startup training failed), so you never need to train it by hand. `POST /train`
retrains in place and is off unless the server is started with
`CREDITLENS_ENABLE_TRAIN=1`. Endpoints:

| Method | Path              | Purpose                                   |
| ------ | ----------------- | ----------------------------------------- |
| GET    | `/health`         | Liveness + whether a model is loaded      |
| POST   | `/train`          | Retrain (`n_samples`, `seed`); `403` by default |
| POST   | `/score`          | Score a single applicant                  |
| POST   | `/score/batch`    | Score a list of applicants                |
| POST   | `/explain/{id}`   | SHAP explanation for one applicant        |
| GET    | `/model/metrics`  | Last training run's evaluation metrics    |
| GET    | `/fairness/report`| Last training run's fairness audit        |

CORS is open by default to support the local frontend demo.

### `frontend/` — Next.js Web Demo
A Next.js 15 + TypeScript app (Tailwind, lucide-react) that provides an
interactive scoring playground: a 22-field applicant form (every input the model
uses; the delinquency-recency field appears once delinquencies > 0), a 0–1000
score rail marking the 450 and 600 cutoffs, a SHAP-style contribution waterfall, and a four-gate fairness
dashboard. It calls the real engine first (`/health`, `POST /score`,
`POST /explain`, `GET /fairness/report`) and renders a **"Live engine"** badge
when the backend answers. Under that badge the score, probability, decision,
waterfall, risk band and fairness metrics are the Python pipeline's, and each shows
nothing until it has answered ("Audit pending" on the fairness card). With no
engine reachable the page runs the engine's recorded model
(`frontend/lib/engineModel.mjs`, `frontend/lib/engineSnapshot.json`).

## Design Principles

1. **Explainability is not optional.** Every score ships with a reason.
2. **Fairness is measured, not assumed — with its uncertainty.** Four
   criteria, each with a confidence interval and a three-state verdict, are
   first-class outputs. They are screening diagnostics, not a legal test.
3. **Reproducible by default.** Seeded data generation and splits make results
   deterministic.
4. **No secrets, no external services.** Everything runs on local synthetic data.

> ⚠️ CreditLens is a research and educational reference implementation. It is
> **not** a production lending system and must not be used to make real credit
> decisions.
