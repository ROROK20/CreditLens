"""The served model has no artificial steps.

Depth-1 trees are a step function on an additive logit: they lump an input's whole
effect at split points. On these features and data, on Linux x86-64, they score the
applicant below 414 / Denied at FICO 487 and 597 / Review at 488, and exactly 414 from FICO 350 to
487, while a strong profile at FICO 350 reads P(default) 0.24 where the label's own
probability is about 0.60. The API serves a sign-constrained
logistic regression, the correct form for this generator's logistic label. These
tests sweep every numeric input in single small steps across its domain, and
compare the response with the label's.
"""

from __future__ import annotations

import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.special import expit, logit

from creditlens.api import app as api
from creditlens.data.domain import (
    BALANCE_TO_MONTHLY_INCOME, INFLOW_TO_MONTHLY_INCOME, MAX_PAYMENT_BURDEN, OUTFLOW_TO_INFLOW, VALIDITY_DOMAIN,
)
from creditlens.data.generator import (
    DatasetConfig, _solve_intercept, default_log_odds, generate_credit_dataset, train_test_split,
)
from creditlens.decision import probability_to_score
from creditlens.features.engineering import FeatureEngineer

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

STEP_APPLICANT = {
    "annual_income": 185887.26, "loan_amount": 11075.46, "loan_term_months": 36, "debt_to_income": 0.23,
    "employment_years": 6.09, "fico_score": 604, "credit_utilization": 0.26, "num_delinquencies": 0,
    "months_since_delinquency": -1, "num_hard_inquiries": 2, "credit_age_months": 141.08,
    "monthly_inflow": 15850.02, "monthly_outflow": 13379.63, "paycheck_regularity": 0.73,
    "bill_payment_consistency": 0.64, "nsf_count": 0, "overdraft_days": 0, "rent_payment_score": 0.81,
    "utility_payment_score": 0.39, "banking_tenure_months": 37.9, "avg_account_balance": 13002.49,
    "savings_rate": 0.13,
}
# The web demo's Strong preset, at the lowest FICO the API accepts.
STRONG_AT_350 = {
    "annual_income": 120000, "loan_amount": 12000, "loan_term_months": 36, "debt_to_income": 0.2,
    "employment_years": 10, "fico_score": 350, "credit_utilization": 0.1, "num_delinquencies": 0,
    "months_since_delinquency": -1, "num_hard_inquiries": 1, "credit_age_months": 84,
    "monthly_inflow": 10000, "monthly_outflow": 7000, "paycheck_regularity": 0.98,
    "bill_payment_consistency": 0.96, "nsf_count": 0, "overdraft_days": 0, "rent_payment_score": 0.95,
    "utility_payment_score": 0.97, "banking_tenure_months": 48, "avg_account_balance": 25000,
    "savings_rate": 0.2,
}

# input -> (one step, how the README writes it). One unit where a unit is small.
STEPS = {
    "fico_score": (1, "1 point"), "debt_to_income": (0.01, "0.01"), "credit_utilization": (0.01, "0.01"),
    "num_delinquencies": (1, "1 more"), "months_since_delinquency": (1, "1 month"),
    "num_hard_inquiries": (1, "1 more"), "credit_age_months": (1, "1 month"), "nsf_count": (1, "1 more"),
    "overdraft_days": (1, "1 day"), "bill_payment_consistency": (0.01, "0.01"),
    "paycheck_regularity": (0.01, "0.01"), "rent_payment_score": (0.01, "0.01"),
    "utility_payment_score": (0.01, "0.01"), "savings_rate": (0.01, "0.01"),
    "banking_tenure_months": (1, "1 month"), "employment_years": (1, "1 year"),
    "avg_account_balance": (100, "$100"), "loan_amount": (100, "$100"), "loan_term_months": (1, "1 month"),
    "monthly_inflow": (100, "$100"), "monthly_outflow": (100, "$100"), "annual_income": (1000, "$1,000"),
}
# One-unit moves of these are whole events; the label's own step is large too.
WHOLE_EVENTS = ("num_delinquencies", "num_hard_inquiries", "nsf_count", "loan_term_months",
                "annual_income", "first_delinquency")
FICO_BOUND = 6
SMALL_STEP_BOUND = 15
N_SWEPT = 200


def _accepted(d: pd.DataFrame) -> np.ndarray:
    """The API's joint rules, vectorised (tests/test_validation.py pins the rules)."""
    monthly = d["annual_income"] / 12
    ratio = d["monthly_inflow"] / monthly
    net = d["monthly_inflow"] - d["monthly_outflow"]
    burden = (d["loan_amount"] / d["loan_term_months"]) / monthly
    lo, hi = VALIDITY_DOMAIN["monthly_net_cashflow"]
    ok = (ratio >= INFLOW_TO_MONTHLY_INCOME[0]) & (ratio <= INFLOW_TO_MONTHLY_INCOME[1])
    ok &= (net >= lo) & (net <= hi) & (burden <= MAX_PAYMENT_BURDEN)
    outflow = d["monthly_outflow"] / d["monthly_inflow"]
    balance = d["avg_account_balance"] / monthly
    ok &= (outflow >= OUTFLOW_TO_INFLOW[0]) & (outflow <= OUTFLOW_TO_INFLOW[1])
    ok &= (balance >= BALANCE_TO_MONTHLY_INCOME[0]) & (balance <= BALANCE_TO_MONTHLY_INCOME[1])
    return ok.to_numpy()


@pytest.fixture(scope="module")
def world():
    """The API's data split, and the label's own intercept from a large draw."""
    df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=api.DEFAULT_SEED))
    train, test = train_test_split(df, test_size=0.20, seed=api.DEFAULT_SEED)
    big = generate_credit_dataset(DatasetConfig(n_samples=200_000, seed=99))
    noise = np.random.default_rng(0).normal(0, 0.30, len(big))
    group = (big["demographic_group"] == "C").to_numpy(dtype=float) * 0.8
    intercept = _solve_intercept(default_log_odds(big) + group + noise, 0.15)
    return train, test, intercept


def _label_p(frame: pd.DataFrame, intercept: float) -> np.ndarray:
    """The label's P(default), averaged over the group mix and the label noise."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(9)
    weights = weights / weights.sum()
    eta = default_log_odds(frame) + intercept
    p = np.zeros(len(frame))
    for share, shift in ((0.85, 0.0), (0.15, 0.8)):
        for z, w in zip(nodes, weights):
            p += share * w * expit(eta + shift + 0.30 * z)
    return p


def _grid(name: str, step: float) -> np.ndarray:
    lo, hi = VALIDITY_DOMAIN[name]
    if name == "num_delinquencies":
        lo = 1
    return np.arange(lo, hi + step / 2, step)


def _largest_step(score, rows: pd.DataFrame, name: str, step: float) -> int:
    """Largest |score change| over adjacent grid values, among rows the API accepts."""
    if name in ("num_delinquencies", "months_since_delinquency"):
        rows = rows[rows["num_delinquencies"] > 0]
    grid = _grid(name, step)
    long = pd.concat([rows.assign(**{name: v}) for v in grid], ignore_index=True)
    scores = np.asarray(score(long), dtype=float).reshape(len(grid), len(rows))
    ok = _accepted(long).reshape(len(grid), len(rows))
    jumps = np.abs(np.diff(scores, axis=0))[ok[1:] & ok[:-1]]
    return int(np.round(jumps.max()))


def _first_delinquency_step(score, rows: pd.DataFrame) -> int:
    """Never delinquent -> one delinquency this month: the count and its recency."""
    rows = rows[rows["num_delinquencies"] == 0]
    moved = rows.assign(num_delinquencies=1, months_since_delinquency=0)
    return int(np.round(np.abs(np.asarray(score(moved), dtype=float) - np.asarray(score(rows), dtype=float)).max()))


def step_table(model, engineer, test, intercept) -> list[tuple[str, str, int, int]]:
    rows = test.iloc[:N_SWEPT]

    def served(frame):
        return probability_to_score(model.predict_proba(engineer.transform(frame)[0])[:, 1])

    def label(frame):
        return 1000 * (1 - _label_p(frame, intercept))

    table = [(name, text, _largest_step(served, rows, name, step), _largest_step(label, rows, name, step))
             for name, (step, text) in STEPS.items()]
    table.append(("first_delinquency", "never to 1, this month",
                  _first_delinquency_step(served, rows), _first_delinquency_step(label, rows)))
    return table


@pytest.fixture(scope="module")
def served(world):
    from creditlens.models.scorer import LogisticScorer

    train, _, _ = world
    engineer = FeatureEngineer(scale=True)
    X, y = engineer.fit_transform(train)
    return LogisticScorer().fit(X, y), engineer


@pytest.fixture(scope="module")
def steps(world, served):
    _, test, intercept = world
    return step_table(*served, test, intercept)


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        api._startup_train()
        yield c


def _readme() -> str:
    return re.sub(r"\s+", " ", README.read_text(encoding="utf-8"))


class TestNoArtificialSteps:
    def test_one_fico_point_does_not_cross_from_denied_to_approved(self, client):
        """Depth-1 trees score 487 at 414 / Denied and 488 at 597 / Review."""
        at = {f: client.post("/score", json={**STEP_APPLICANT, "fico_score": f}).json()
              for f in (487, 488)}
        assert abs(at[488]["credit_score"] - at[487]["credit_score"]) <= FICO_BOUND, at

    def test_fico_has_no_plateau(self, client):
        """Depth-1 trees score 350 to 487 all at 414, so 137 points of FICO buy nothing.
        The served score moves with every point (rounding aside) all the way down."""
        scores = [client.post("/score", json={**STEP_APPLICANT, "fico_score": f}).json()["credit_score"]
                  for f in range(350, 488)]
        assert scores == sorted(scores)
        assert scores[-1] - scores[0] > 300, (scores[0], scores[-1])
        assert all(b > a for a, b in zip(scores[::10], scores[10::10])), scores[::10]

    def test_strong_profile_at_fico_350_reads_the_labels_probability(self, client, world):
        _, _, intercept = world
        served = client.post("/score", json=STRONG_AT_350).json()["default_probability"]
        frame = pd.DataFrame([STRONG_AT_350])
        true = float(_label_p(frame, intercept)[0])
        assert abs(logit(served) - logit(true)) < 0.25, (served, true)
        assert f"a strong profile at FICO 350 reads P(default) {served:.2f} against the label's {true:.2f}" in _readme()


# The label's weights in the model's own units (data/generator.py default_log_odds).
LABEL_WEIGHTS = {
    "fico_score": -0.02, "debt_to_income": 3.5, "credit_utilization": 2.5, "num_delinquencies": 0.9,
    "months_since_delinquency": -1.2 / 24, "num_hard_inquiries": 0.15, "credit_age_months": -0.003,
    "nsf_count": 0.35, "overdraft_days": 0.02, "bill_payment_consistency": -1.2, "paycheck_regularity": -1.0,
    "rent_payment_score": -1.2, "utility_payment_score": -0.8, "monthly_net_cashflow": -0.00015,
    "savings_rate": -2.5, "avg_account_balance": -0.2, "banking_tenure_months": -0.004,
    "employment_years": -0.03, "monthly_payment_burden": 2.0,
}


def test_the_fitted_weights_sit_within_two_standard_errors_of_the_labels(world, served):
    """Why an input's ratio can read high: its estimate carries sampling error. The
    README says so and quotes the worst case; this is the measurement behind it."""
    train, _, _ = world
    model, engineer = served
    X = engineer.transform(train)[0]
    Z = np.column_stack([np.ones(len(X)), X.to_numpy()])
    p = model.predict_proba(X)[:, 1]
    errors = np.sqrt(np.diag(np.linalg.inv((Z * (p * (1 - p))[:, None]).T @ Z)))[1:]
    assert set(LABEL_WEIGHTS) == set(X.columns)
    scale = pd.Series(engineer._scaler.scale_, index=X.columns)
    weight = model.coefficients() / scale
    error = pd.Series(errors, index=X.columns) / scale
    for name, true in LABEL_WEIGHTS.items():
        assert abs(weight[name] - true) <= 2 * error[name], (name, weight[name], true, error[name])
    assert (f"(0.02 log-odds a day, estimated at {weight['overdraft_days']:.3f} with a standard error of "
            f"{error['overdraft_days']:.3f} on 8,000 rows)") in _readme()


class TestEverySmallStepIsBounded:
    def test_every_numeric_input_is_swept(self, steps):
        from creditlens.features.engineering import MODEL_INPUTS

        assert {r[0] for r in steps} - {"first_delinquency"} == set(MODEL_INPUTS)

    def test_the_bounds_hold(self, steps):
        by_name = {r[0]: r for r in steps}
        assert by_name["fico_score"][2] <= FICO_BOUND
        for name, _, model, truth in steps:
            if name not in WHOLE_EVENTS:
                assert model <= SMALL_STEP_BOUND, (name, model)
            # A whole event is large because the label's is; the response tracks it.
            assert model <= 1.3 * truth + 6, (name, model, truth)

    def test_the_readme_publishes_the_table_and_the_bound(self, steps):
        text = README.read_text(encoding="utf-8")
        published = re.findall(r"^\| `(\w+)` \| ([^|]+) \| (\d+) \| (\d+) \|$", text, re.M)
        assert [(n, s, int(m), int(t)) for n, s, m, t in published] == steps
        worst = max(steps, key=lambda r: r[2])
        small = max((r for r in steps if r[0] not in WHOLE_EVENTS), key=lambda r: r[2])
        prose = _readme()
        assert (f"no FICO point moves a score more than {FICO_BOUND}, and no small step "
                f"more than {SMALL_STEP_BOUND}") in prose
        assert f"(the largest, `{small[0]}`, is {small[2]})" in prose
        assert f"the largest single step anywhere is `{worst[0]}`, {worst[2]} points" in prose


STUMPS = dict(n_estimators=400, max_depth=1, learning_rate=0.1, calibrate=False, scale_pos_weight=1.0)

# XGBoost at the same pinned version trains slightly differently across platforms, so
# its published figures are exact only where they were measured.
PLATFORM = (platform.system(), platform.machine())
CI_PLATFORM = ("Linux", "x86_64")
MAC_ARM = ("Darwin", "arm64")


def comparison_row(seed: int, make) -> tuple[float, float, int, int, int]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    from tests.test_generative_links import PROPORTIONALITY_CASES, _response

    df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=seed))
    train, test = train_test_split(df, test_size=0.20, seed=seed)
    engineer = FeatureEngineer(scale=True)
    X, y = engineer.fit_transform(train)
    X_test, y_test = engineer.transform(test)
    model = make().fit(X, y)
    p = model.predict_proba(X_test)[:, 1]
    responses = [_response(name, model, engineer, train, test) for name in PROPORTIONALITY_CASES]

    def score(frame):
        return probability_to_score(model.predict_proba(engineer.transform(frame)[0])[:, 1])

    rows = test.iloc[:N_SWEPT]
    return (round(float(roc_auc_score(y_test, p)), 4), round(float(brier_score_loss(y_test, p)), 4),
            sum(r["unjustified"] for r in responses), sum(r["flips"] for r in responses),
            _largest_step(score, rows, "fico_score", 1))


def test_the_readme_publishes_the_comparison_with_stumps():
    """Why the API serves logistic regression, re-measured."""
    from creditlens.models.scorer import LogisticScorer, XGBoostScorer

    published = re.findall(
        r"^\| (\d+) \| (logistic|XGBoost stumps) \| ([\d.]+) \| ([\d.]+) \| (\d+) of (\d+) \(([\d.]+)%\) \| (\d+) \|$",
        README.read_text(encoding="utf-8"), re.M)
    assert len(published) == 6
    by_seed: dict[str, dict] = {}
    for seed, name, auc, brier, unjustified, flips, pct, jump in published:
        make = LogisticScorer if name == "logistic" else (lambda: XGBoostScorer(**STUMPS))
        row = comparison_row(int(seed), make)
        if name == "logistic" or PLATFORM == CI_PLATFORM:
            assert row == (float(auc), float(brier), int(unjustified), int(flips), int(jump)), (seed, name, row)
        assert f"{int(unjustified) / int(flips):.1%}" == f"{pct}%"
        by_seed.setdefault(seed, {})[name] = row
    # The conclusion is checked on what was just trained, so it holds on any platform.
    for seed, rows in by_seed.items():
        lr, stumps = rows["logistic"], rows["XGBoost stumps"]
        assert lr[0] > stumps[0] and lr[1] < stumps[1], (seed, lr, stumps)
        assert lr[2] / lr[3] < stumps[2] / stumps[3] / 2 and lr[4] < stumps[4], (seed, lr, stumps)
    if PLATFORM == MAC_ARM:
        assert by_seed["42"]["XGBoost stumps"] == (0.8514, 0.0926, 74, 656, 197)
        assert "seed 42 reads AUC 0.8514, 74 of 656 flips and a 197-point jump" in _readme()
    jumps = [r["XGBoost stumps"][4] for r in by_seed.values()]
    if PLATFORM == CI_PLATFORM:
        assert f"at most 6 points against {min(jumps)} to {max(jumps)}." in _readme()


def test_the_readme_quotes_what_the_stumps_do_to_one_applicant():
    from creditlens.decision import score_to_decision
    from creditlens.models.scorer import XGBoostScorer

    df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=api.DEFAULT_SEED))
    train, _ = train_test_split(df, test_size=0.20, seed=api.DEFAULT_SEED)
    engineer = FeatureEngineer(scale=True)
    X, y = engineer.fit_transform(train)
    model = XGBoostScorer(**STUMPS).fit(X, y)

    def probability(applicant):
        return model.predict_proba(engineer.transform(pd.DataFrame([applicant]))[0])[:, 1]

    def score(fico):
        return int(probability_to_score(probability({**STEP_APPLICANT, "fico_score": fico}))[0])

    prose = _readme()
    below, above = score(487), score(488)
    assert score(350) == below
    assert f"reads P(default) {float(probability(STRONG_AT_350)[0]):.2f} where the label's own" in prose
    if PLATFORM == CI_PLATFORM:
        assert (f"{below} / {score_to_decision(below)} at FICO 487 and {above} / {score_to_decision(above)} at 488, "
                f"and {below} at every FICO from 350 to 487") in prose
    elif PLATFORM == MAC_ARM:
        assert f"{below} / {score_to_decision(below)} to {above} / {score_to_decision(above)}" in prose


def test_a_step_function_is_what_the_stumps_still_do():
    """The tree model stays in the library, so what it does stays measured: the
    comparison above is against a model that has this step."""
    from creditlens.models.scorer import XGBoostScorer

    row = comparison_row(api.DEFAULT_SEED, lambda: XGBoostScorer(**STUMPS))
    assert row[4] > 100
