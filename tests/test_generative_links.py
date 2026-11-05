"""Every model input has a documented effect on the synthetic default label, in
the direction the model is constrained to, and the model's response tracks it.

The ground truth is data/generator.py default_log_odds(); these tests perturb
each field through it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from creditlens.api.app import ApplicantFeatures
from creditlens.data.generator import (
    LABEL_EFFECTS, DatasetConfig, default_log_odds, generate_credit_dataset,
)
from creditlens.features.engineering import (
    INPUT_RISK_DIRECTION, MODEL_INPUTS, NUMERIC_FEATURES, RISK_DIRECTION, FeatureEngineer,
)

RAW_MODEL_INPUTS = sorted(MODEL_INPUTS)
DERIVED = sorted(set(NUMERIC_FEATURES) - set(RAW_MODEL_INPUTS))
NO_LINK = ("num_open_accounts", "num_subscriptions", "loan_purpose", "interest_rate")

# A small increase per field (in its own units), kept inside the generated range.
STEP = {"fico_score": 5, "num_delinquencies": 1, "months_since_delinquency": 1,
        "num_hard_inquiries": 1, "nsf_count": 1, "overdraft_days": 1, "loan_term_months": 12,
        "credit_age_months": 6, "banking_tenure_months": 6, "employment_years": 1}


@pytest.fixture(scope="module")
def applicants():
    df = generate_credit_dataset(DatasetConfig(n_samples=400, seed=11))
    return df.drop(columns=["default", "demographic_group"])


def _effect(frame: pd.DataFrame, field: str) -> np.ndarray:
    step = STEP.get(field, 0.1 * float(frame[field].std()) or 0.01)
    moved = frame.copy()
    if field == "months_since_delinquency":
        moved = moved[(moved["num_delinquencies"] > 0) & (moved["months_since_delinquency"] < 23)]
        frame = frame.loc[moved.index]
        moved[field] = moved[field] + step
    else:
        moved[field] = moved[field] + step
    return default_log_odds(moved) - default_log_odds(frame)


def test_label_effects_document_exactly_the_model_inputs():
    assert set(LABEL_EFFECTS) == set(RAW_MODEL_INPUTS)
    assert set(RAW_MODEL_INPUTS) <= set(ApplicantFeatures.model_fields)


def test_every_model_feature_is_one_term_of_the_label():
    """Each input reaches the model by ONE path: the features are the label's terms,
    each once, so the model cannot learn income as a proxy beside the burden it acts
    through."""
    terms = {name for name in RAW_MODEL_INPUTS if not LABEL_EFFECTS[name].startswith("via ")}
    assert set(NUMERIC_FEATURES) == terms | {"monthly_payment_burden", "monthly_net_cashflow"}
    assert len(NUMERIC_FEATURES) == len(set(NUMERIC_FEATURES)) == 19
    assert set(RISK_DIRECTION) == set(NUMERIC_FEATURES)


@pytest.mark.parametrize("field", RAW_MODEL_INPUTS)
def test_every_model_input_moves_true_risk_in_its_constrained_direction(applicants, field):
    delta = _effect(applicants, field)
    moved = delta[np.abs(delta) > 1e-12]
    assert len(moved) > 0, f"{field} has no effect on the synthetic default label"
    assert (np.sign(moved) == INPUT_RISK_DIRECTION[field]).all(), (field, INPUT_RISK_DIRECTION[field])


@pytest.mark.parametrize("field", NO_LINK)
def test_fields_without_a_link_have_no_effect_and_are_not_inputs(applicants, field):
    moved = applicants.copy()
    moved[field] = "other" if field == "loan_purpose" else moved[field] + 5
    np.testing.assert_array_equal(default_log_odds(moved), default_log_odds(applicants))
    assert field not in NUMERIC_FEATURES and field not in ApplicantFeatures.model_fields


@pytest.mark.parametrize("derived", DERIVED)
def test_derived_features_agree_with_every_input_that_feeds_them(applicants, derived):
    """A derived feature's constraint must not fight its inputs: raising an
    input must move the derived feature in the direction that changes risk the
    same way as the input itself."""
    engineer = FeatureEngineer(scale=False)
    engineer.fit_transform(pd.concat([applicants, pd.Series(0, index=applicants.index, name="default")], axis=1))
    base = engineer._add_derived_features(applicants.copy())[derived]
    feeds = 0
    for field in RAW_MODEL_INPUTS:
        if field == "months_since_delinquency":
            continue
        moved = applicants.copy()
        moved[field] = moved[field] + STEP.get(field, 0.1 * float(applicants[field].std()))
        change = engineer._add_derived_features(moved)[derived] - base
        change = change[np.abs(change) > 1e-12]
        if len(change):
            feeds += 1
            assert (np.sign(change) * RISK_DIRECTION[derived] == INPUT_RISK_DIRECTION[field]).all(), (derived, field)
    assert feeds > 0


def test_recency_beyond_the_plateau_buys_nothing():
    """The label's recency risk is flat from 24 months; the model
    feature is capped there, so an older delinquency cannot earn points."""
    from fastapi.testclient import TestClient

    from creditlens.api.app import app

    applicant = {
        "annual_income": 75000, "loan_amount": 12000, "loan_term_months": 36, "fico_score": 680,
        "debt_to_income": 0.32, "employment_years": 4, "credit_utilization": 0.45,
        "credit_age_months": 84, "num_hard_inquiries": 2, "num_delinquencies": 2,
        "monthly_inflow": 6250, "monthly_outflow": 4800, "avg_account_balance": 8500,
        "paycheck_regularity": 0.8, "bill_payment_consistency": 0.8, "nsf_count": 0,
        "overdraft_days": 0, "rent_payment_score": 0.8, "utility_payment_score": 0.8,
        "banking_tenure_months": 12, "savings_rate": 0.1,
    }
    with TestClient(app) as c:
        scores = {m: c.post("/score", json={**applicant, "months_since_delinquency": m}).json()["credit_score"]
                  for m in (24, 36, 60, 119, 120)}
        assert len(set(scores.values())) == 1, scores
        assert c.post("/score", json={**applicant, "months_since_delinquency": 121}).status_code == 422


# --- proportionality: every input, against the generator's true effect -------
# Direction is not enough: the served model must not over-reward an input.
# For every not-approved test applicant, improve ONE input (or one group the
# API couples) by about one training standard deviation, inside the training
# range, and compare the model's change in log-odds with the generator's true
# change (default_log_odds). A flip to Approved is "unjustified" when the true
# change, applied to the applicant's served log-odds, would not cross the
# approval cutoff. README "Model behaviour" publishes the table this computes,
# and must name every input that misses a bar: it may not claim proportionality
# for an input the measurement contradicts.

IN_PROPORTION = (0.6, 1.4)  # model change / true change
UNJUSTIFIED_BAR = 0.20  # of the flips to Approved
INTEGER_INPUTS = ("fico_score", "num_hard_inquiries", "nsf_count", "overdraft_days")


def _improve(name: str, d: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    sd, lo, hi = train.std(numeric_only=True), train.min(numeric_only=True), train.max(numeric_only=True)
    up = ("fico_score", "credit_age_months", "bill_payment_consistency", "paycheck_regularity",
          "rent_payment_score", "utility_payment_score", "savings_rate",
          "banking_tenure_months", "employment_years")
    down = ("debt_to_income", "credit_utilization", "num_hard_inquiries", "nsf_count",
            "overdraft_days", "loan_amount")
    d = d.copy()
    if name in up:
        d[name] = (d[name] + sd[name]).clip(upper=hi[name])
    elif name in down:
        d[name] = (d[name] - sd[name]).clip(lower=lo[name])
    elif name == "avg_account_balance":  # +1 SD, inside the training ratio to income (3.0x)
        d[name] = np.maximum(d[name], np.minimum(d[name] + sd[name], 3.0 * d["annual_income"] / 12))
    elif name == "loan_term_months":
        d[name] = (d[name] + 12).clip(upper=hi[name])
    elif name == "months_since_delinquency":
        late = d["num_delinquencies"] > 0
        d.loc[late, name] = (d.loc[late, name] + 12).clip(upper=hi[name])
    elif name == "delinquency_history":  # one fewer delinquency, recency with it
        d["months_since_delinquency"] = np.where(d["num_delinquencies"] <= 1, -1,
                                                 d["months_since_delinquency"])
        d["num_delinquencies"] = (d["num_delinquencies"] - 1).clip(lower=0)
    elif name == "monthly_inflow":  # +10%, inside the training ratios to income and to outflow
        raised = np.minimum(d[name] * 1.1, np.minimum(1.2 * d["annual_income"] / 12, d["monthly_outflow"] / 0.50))
        d[name] = np.maximum(d[name], raised)
    elif name == "monthly_outflow":
        d[name] = np.minimum(d[name], (d[name] - sd[name]).clip(lower=np.maximum(lo[name], 0.5 * d["monthly_inflow"])))
    elif name == "annual_income":  # +20% with the rest unchanged, inside the 0.80x and 0.5x ratios
        raised = np.minimum(np.minimum(d[name] * 1.2, d["monthly_inflow"] * 12 / 0.80),
                            d["avg_account_balance"] * 12 / 0.5)
        d[name] = np.maximum(d[name], raised)
    elif name == "income_cash_flow_and_balance":  # x2, everything it bounds with it
        scale = np.minimum(2.0, hi["annual_income"] / d["annual_income"])
        for col in COUPLED[name]:
            d[col] = d[col] * scale
    elif name == "rent_and_utility_to_1":  # the scenario that first exposed over-learning
        d["rent_payment_score"] = 1.0
        d["utility_payment_score"] = 1.0
    elif name == "stated_income_x2":  # the flows unchanged: a 422 at the API, see below
        d["annual_income"] = d["annual_income"] * np.minimum(2.0, hi["annual_income"] / d["annual_income"])
    else:
        raise KeyError(name)
    for col in INTEGER_INPUTS:
        d[col] = d[col].round()
    return d


PROPORTIONALITY_CASES = (
    "fico_score", "debt_to_income", "credit_utilization", "delinquency_history",
    "months_since_delinquency", "num_hard_inquiries", "credit_age_months",
    "nsf_count", "overdraft_days", "bill_payment_consistency", "paycheck_regularity",
    "rent_payment_score", "utility_payment_score", "savings_rate",
    "avg_account_balance", "banking_tenure_months", "employment_years",
    "loan_amount", "loan_term_months", "monthly_inflow", "monthly_outflow",
    "annual_income", "income_cash_flow_and_balance",
)
COUPLED = {"delinquency_history": ("num_delinquencies", "months_since_delinquency"),
           "income_cash_flow_and_balance": ("annual_income", "monthly_inflow", "monthly_outflow",
                                            "avg_account_balance")}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _served_setup(seed: int | None = None):
    """The model the API serves, trained by the API's own code path, with the
    data it was trained on. A non-default seed retrains the same configuration."""
    from creditlens.api import app as api
    from creditlens.data.generator import train_test_split

    seed = api.DEFAULT_SEED if seed is None else seed
    api.train_model(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=seed)
    df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=seed))
    train, test = train_test_split(df, test_size=0.20, seed=seed)
    return api._model_state["model"], api._model_state["engineer"], train, test


def _response(case: str, model, engineer, train, test) -> dict:
    from creditlens.decision import probability_to_score

    p0 = model.predict_proba(engineer.transform(test)[0])[:, 1]
    not_approved = probability_to_score(p0) < 600
    moved = _improve(case, test, train)
    p1 = model.predict_proba(engineer.transform(moved)[0])[:, 1]
    true = default_log_odds(moved) - default_log_odds(test)
    change = _logit(p1) - _logit(p0)
    flips = not_approved & (probability_to_score(p1) >= 600)
    justified = not_approved & (probability_to_score(1 / (1 + np.exp(-(_logit(p0) + true)))) >= 600)
    return {"ratio": float(np.mean(change[not_approved]) / np.mean(true[not_approved])),
            "excess": float(np.mean(np.abs(change[not_approved])) - np.mean(np.abs(true[not_approved]))),
            "median_ratio": float(np.median(change[not_approved]) / np.median(true[not_approved])),
            "flips": int(flips.sum()), "unjustified": int((flips & ~justified).sum())}


def proportionality_table(seed: int | None = None) -> list[tuple[str, float, int, int]]:
    """(case, model/true log-odds ratio, flips to Approved, unjustified flips)
    on the served model's not-approved test applicants."""
    setup = _served_setup(seed)
    rows = []
    for name in PROPORTIONALITY_CASES:
        r = _response(name, *setup)
        rows.append((name, round(r["ratio"], 2), r["flips"], r["unjustified"]))
    return rows


@pytest.fixture(scope="module")
def proportionality():
    return proportionality_table()


@pytest.fixture()
def restore_startup_model():
    from creditlens.api import app as api

    yield
    api._startup_train()


def test_proportionality_covers_every_input():
    covered = set()
    for name in PROPORTIONALITY_CASES:
        covered |= set(COUPLED.get(name, (name,)))
    assert covered == set(RAW_MODEL_INPUTS)


def test_every_proportionality_move_is_an_applicant_the_api_accepts():
    """The table measures responses the API can actually serve: each moved row
    passes the request validators (domain and joint rules) unchanged."""
    from creditlens.api import app as api
    from creditlens.data.generator import train_test_split
    from creditlens.data.domain import INTEGER_FIELDS

    df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=api.DEFAULT_SEED))
    train, test = train_test_split(df, test_size=0.20, seed=api.DEFAULT_SEED)
    fields = [f for f in ApplicantFeatures.model_fields if f != "demographic_group"]
    for name in PROPORTIONALITY_CASES:
        moved = _improve(name, test, train)
        for _, row in moved[fields].iloc[:200].iterrows():
            values = {k: int(v) if k in INTEGER_FIELDS else float(v) for k, v in row.items()}
            ApplicantFeatures(**values)  # raises on anything the API would refuse


def test_no_input_is_over_rewarded_and_every_input_still_responds(proportionality):
    """A ratio is a poor bar for the smallest effect in the label (overdraft_days, 0.02
    per day, whose estimate carries a standard error of 0.027 on 8,000 rows), so an input
    over 1.4x must be named with its figure (test below) and its absolute error bounded
    here: the model's change may exceed the true change by at most 0.1 log-odds on the
    published move. An input must still move the model."""
    setup = _served_setup()
    for name, ratio, _, _ in proportionality:
        assert 0.25 < ratio, (name, ratio)
        if ratio >= IN_PROPORTION[1]:
            model, engineer, train, test = setup
            excess = _response(name, model, engineer, train, test)["excess"]
            assert excess < 0.1, (name, ratio, excess)


def test_readme_names_the_largest_response(proportionality):
    name, ratio, _, _ = max(proportionality, key=lambda r: r[1])
    assert f"the largest response is `{name}`, {ratio:.2f}x" in re.sub(r"\s+", " ", _readme())


def test_the_model_clears_the_unjustified_flip_bar_overall(proportionality):
    flips = sum(f for _, _, f, _ in proportionality)
    unjustified = sum(u for _, _, _, u in proportionality)
    assert flips > 300 and unjustified <= UNJUSTIFIED_BAR * flips, (unjustified, flips)


@pytest.mark.parametrize("seed", (7, 123))
def test_the_overall_bar_holds_on_seeds_the_table_was_not_measured_on(seed, restore_startup_model):
    """The API serves seed 42, so that is the published table; these two seeds
    show the bar is not an accident of it."""
    rows = proportionality_table(seed)
    flips, unjustified = sum(r[2] for r in rows), sum(r[3] for r in rows)
    assert flips > 300 and unjustified <= UNJUSTIFIED_BAR * flips, (seed, unjustified, flips)


def test_doubling_stated_income_is_no_longer_a_proxy(restore_startup_model):
    """The original finding, measured the original way: x2 stated income with
    the cash flows unchanged moved the log-odds 2.15x its true effect (median
    over the not-approved) and flipped 47 to Approved, 30 unjustified. It is
    now in proportion, and the API refuses that input outright. (The bound was
    1.0 while trees under-learned the burden; the logistic weight sits within
    its standard error of the true one, which reads as 1.08x here.)"""
    r = _response("stated_income_x2", *_served_setup())
    assert r["median_ratio"] < IN_PROPORTION[1], r
    assert r["flips"] < 47 and r["unjustified"] < 30, r

    from fastapi.testclient import TestClient

    from creditlens.api.app import app

    applicant = {"annual_income": 150000, "loan_amount": 12000, "loan_term_months": 36,
                 "debt_to_income": 0.32, "employment_years": 4, "fico_score": 720,
                 "credit_utilization": 0.25, "num_delinquencies": 0,
                 "months_since_delinquency": -1, "num_hard_inquiries": 1, "credit_age_months": 84,
                 "monthly_inflow": 6250, "monthly_outflow": 4800, "paycheck_regularity": 0.92,
                 "bill_payment_consistency": 0.88, "nsf_count": 0, "overdraft_days": 0,
                 "rent_payment_score": 0.95, "utility_payment_score": 0.85,
                 "banking_tenure_months": 48, "avg_account_balance": 8500, "savings_rate": 0.1}
    with TestClient(app) as c:
        assert c.post("/score", json=applicant).status_code == 422


def test_payment_behaviour_is_not_over_rewarded(restore_startup_model):
    """Rent and utility to 1.0 for every not-approved audit applicant: the
    scenario that first exposed over-learning (1.84x at 4,000 training rows, 24
    of 51 flips unjustified). The lower bound, relaxed to 0.25 while depth-1 trees
    under-learned these two (0.43x), is 0.6 again on the logistic model."""
    r = _response("rent_and_utility_to_1", *_served_setup())
    assert IN_PROPORTION[0] < r["median_ratio"] < IN_PROPORTION[1], r
    assert r["unjustified"] <= UNJUSTIFIED_BAR * r["flips"], r


def _readme() -> str:
    return (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")


def test_readme_publishes_the_measured_proportionality_table(proportionality):
    published = re.findall(r"^\| `(\w+)` \| [^|]+ \| ([\d.]+) \| (\d+) \| (\d+) \|$", _readme(), re.M)
    assert [(n, float(r), int(f), int(u)) for n, r, f, u in published] == proportionality


def test_readme_names_every_input_that_misses_a_bar(proportionality):
    """An input outside the band, or over the unjustified-flip bar, is named
    with its measured figure. The README may not claim proportionality for it."""
    text = re.sub(r"\s+", " ", _readme())

    def clause(start: str) -> str:
        rest = text[text.index(start):]
        return rest[:rest.index(". ")]

    assert re.findall(r"`(\w+)` ([\d.]+)", clause("Over 1.4x:")) == [
        (n, f"{r:.2f}") for n, r, _, _ in proportionality if r >= IN_PROPORTION[1]]
    assert re.findall(r"`(\w+)` ([\d.]+)", clause("Under-learned (ratio below 0.6):")) == [
        (n, f"{r:.2f}") for n, r, _, _ in proportionality if r < IN_PROPORTION[0]]
    assert re.findall(r"`(\w+)` (\d+) of (\d+)", clause("Over the 20% bar on their own:")) == [
        (n, str(u), str(f)) for n, _, f, u in proportionality if u > UNJUSTIFIED_BAR * f]
    flips = sum(f for _, _, f, _ in proportionality)
    unjustified = sum(u for _, _, _, u in proportionality)
    assert f"{unjustified} of the {flips} flips ({unjustified / flips:.1%}) are unjustified" in text


# Seeds the proportionality claims are stated over: 1-50 and 123. Each run re-derives the
# two the table compares with, the first ten, and every seed behind a stated extreme (43,
# and the zero-weight seeds past 10); CREDITLENS_FULL_SEED_SWEEP=1 re-runs the whole window.
SEEDS_STATED = (42, 7, 123, 1, 2, 3, 4, 5, 6, 8, 9, 10, 22, 23, 26, 38, 43, 47)
if os.environ.get("CREDITLENS_FULL_SEED_SWEEP"):
    SEEDS_STATED = (*range(1, 51), 123)


def test_readme_states_the_other_seeds(restore_startup_model):
    text = re.sub(r"\s+", " ", _readme())
    tables = {seed: proportionality_table(seed) for seed in SEEDS_STATED}
    for seed in (7, 123):
        rows = tables[seed]
        flips, unjustified = sum(r[2] for r in rows), sum(r[3] for r in rows)
        assert f"seed {seed}: {unjustified} of {flips} ({unjustified / flips:.1%})" in text, seed
    overdraft = {seed: next(r[1] for r in rows if r[0] == "overdraft_days") for seed, rows in tables.items()}
    others = [(r[1], r[0], seed) for seed, rows in tables.items() for r in rows if r[0] != "overdraft_days"]
    ratio, name, seed = max(others)
    top = max(overdraft, key=overdraft.get)
    assert (f"`overdraft_days` runs from {min(overdraft.values()) + 0.0:.2f}x to {overdraft[top]:.2f}x "
            f"(seed {top}), and no other input exceeds {ratio:.2f}x (`{name}`, seed {seed})") in text
    zero = sorted(seed for seed, r in overdraft.items() if abs(r) < 0.005)
    assert f"on seeds {', '.join(map(str, zero))} its weight is exactly 0" in text
    worst = max(((sum(r[3] for r in rows) / sum(r[2] for r in rows), seed) for seed, rows in tables.items()))
    assert f"at most {worst[0]:.1%} unjustified (seed {worst[1]}), under the 20% bar" in text
    assert worst[0] < 0.20
    assert "Over seeds 1–50 and 123" in text


def test_readme_states_the_rent_and_utility_and_stated_income_responses(restore_startup_model):
    text = re.sub(r"\s+", " ", _readme())
    setup = _served_setup()
    rent = _response("rent_and_utility_to_1", *setup)
    assert (f"Setting rent and utility to 1.0 together moves the log-odds {rent['ratio']:.2f}x the true "
            f"change ({rent['flips']} flips, {rent['unjustified']} unjustified)") in text
    income = _response("stated_income_x2", *setup)
    assert f"the model's own response to it is {income['median_ratio']:.2f}x (median)" in text
