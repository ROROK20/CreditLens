# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a vulnerability

Please do not report vulnerabilities through public GitHub issues, discussions or pull
requests. Email **rohan.s.khiani.vora@gmail.com** with a description and its impact,
steps to reproduce, the affected version and component, and any suggested fix.

We aim to acknowledge a report within 48 hours and triage it within 7 days, then
coordinate the fix and its disclosure with you, crediting you unless you prefer
otherwise. Testing a local instance is always welcome. CreditLens runs on synthetic data
only and needs no secrets; it is a reference implementation, not a production
credit-decisioning system.

## Dependency audit (2026-11-04)

| Set | Reproduce | Result |
| --- | --------- | ------ |
| Python, exact pins | `pip-audit -r requirements.txt` (pip-audit 2.10.1) | No known vulnerabilities |
| Frontend, what ships | `cd frontend && npm audit --omit=dev` (Node 23.9) | 0 vulnerabilities |
| Frontend, with dev tools | `cd frontend && npm audit` (Node 23.9) | 0 critical, 0 high, 6 moderate |

The 6 moderate rows are one advisory, `minimatch` GHSA-g3hv-75wg-m249, through
`@typescript-eslint`: lint only, never bundled. `npm audit fix` is not applied, so the
lockfile the suite runs on is the one audited. `frontend/package.json` overrides Next's
own `postcss` pin with the frontend's `postcss` 8.5.32, which clears that pin's high
advisories.

Advisory databases move faster than this file, so re-run these rather than trusting it.
Exact pins cover the declared dependencies, not a full lockfile: transitive packages
(`starlette`, `numba`) float within their parents' ranges, so use a hash-pinned lockfile
in a real deployment, and run `pip install --upgrade pip` in a fresh virtual environment.

## Input validation

`tests/test_validation.py` checks every bound and rule here against the live schema.

- **`413`** for any request body over 2 MiB (2,097,152 bytes), before it is parsed,
  whether it declares a `Content-Length` or arrives chunked. The largest valid request
  (1,000 applicants, every field at full float precision) is 865,016 bytes (measured
  2026-11-04).
- **`400`** for a body that cannot be parsed at all: a 5,000-digit integer or 100,000 open
  brackets, which Python's JSON parser refuses.
- **`422`** for anything else invalid, on `POST /score`, `/score/batch`,
  `/explain/{applicant_idx}` and `/train`:
  - numbers must be finite, and a JSON string holding a lone surrogate (`"\ud800"`) is
    refused wherever it appears;
  - types are strict: `"fico_score": "700"`, `"nsf_count": true` and `"fico_score": 700.0`
    are rejected (a JSON integer is accepted for a decimal field);
  - unknown fields are rejected and every model input is required. Nothing is imputed, and
    the library agrees: `FeatureEngineer.transform` raises on an absent or NaN input unless
    the caller passes `impute_missing=True`, which warns. Only `demographic_group` may be
    left out, and it is ignored;
  - `monthly_net_cashflow`, `interest_rate`, `loan_purpose`, `num_open_accounts` and
    `num_subscriptions` are not inputs, so sending one is a `422` like any unknown field;
  - every applicant field must lie in the range below, inclusive: the default training
    data's observed range widened by 5% of that range (on a log scale for the five dollar
    amounts) and rounded outward (`creditlens/data/domain.py`);
  - five joint rules hold: `monthly_inflow` is 0.78 to 1.22 x `annual_income` / 12,
    `monthly_outflow` is 0.47 to 0.98 x `monthly_inflow`, `avg_account_balance` is 0.37 to
    3.13 x `annual_income` / 12, the monthly payment `loan_amount / loan_term_months` is at
    most 2.8 x monthly income, and `monthly_inflow - monthly_outflow` lies in -1,200 to
    25,000;
  - `months_since_delinquency` is `-1` ("never delinquent") exactly when
    `num_delinquencies` is `0`.

| Applicant field | Accepted |
|---|---|
| `annual_income` | 12,000 to 600,000 |
| `loan_amount` | 830 to 49,000 |
| `loan_term_months` | 9 to 63 |
| `debt_to_income` | 0.01 to 0.74 |
| `employment_years` | 0 to 21 |
| `fico_score` | 350 to 850 |
| `credit_utilization` | 0 to 0.98 |
| `num_delinquencies` | 0 to 5 |
| `months_since_delinquency` | -1 to 120 |
| `num_hard_inquiries` | 0 to 8 |
| `credit_age_months` | 0 to 378 |
| `monthly_inflow` | 820 to 61,000 |
| `monthly_outflow` | 410 to 57,000 |
| `paycheck_regularity` | 0 to 1 |
| `bill_payment_consistency` | 0.05 to 1 |
| `nsf_count` | 0 to 6 |
| `overdraft_days` | 0 to 10 |
| `rent_payment_score` | 0.13 to 1 |
| `utility_payment_score` | 0.06 to 1 |
| `banking_tenure_months` | 0 to 252 |
| `avg_account_balance` | 480 to 170,000 |
| `savings_rate` | 0 to 0.38 |
| `demographic_group` | text, at most 64 characters, or omitted |

| `POST /train` field | Accepted |
|---|---|
| `n_samples` | 2,000 to 50,000, or omitted |
| `seed` | 0 to 4,294,967,295, or omitted |

| `POST /score/batch` field | Accepted |
|---|---|
| `applicants` | 1 to 1,000 applicants |

## What a deployer needs to know

- **Inputs are unverified.** Every input is self-reported, so an applicant can claim
  favourable cash flow inside these rules and move a decision: a lender must verify cash
  flow independently (README "Model behaviour" measures it). Inside the domain, a value or
  ratio outside the training data's central 99% is scored and named in `warnings`, and so
  are two or more cash-flow ratios in their most favourable tenth at once.
- **`POST /train` is off by default** (`403`). It is unauthenticated and replaces the model
  for every later caller, so it answers only when the server is started with
  `CREDITLENS_ENABLE_TRAIN=1`; then anyone who can reach the API can change what later
  requests return and spend CPU doing it.
- **No authentication, no rate limiting.** The scoring endpoints persist nothing. The bounds
  stop malformed and absurd input; they do not limit traffic. Add an API key and rate
  limiting, and serve a pre-fitted model, before exposing the API publicly at scale.
- **CORS is `*`** for this credential-free demo. Restrict `allow_origins` to your frontend's
  origin in any other deployment.
- **Nothing sensitive is stored.** `demographic_group` is never a model feature and never
  stored. `.env` is git-ignored, and only `.env.example` placeholders are tracked.
