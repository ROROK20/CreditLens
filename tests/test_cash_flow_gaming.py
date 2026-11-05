"""Stated cash flow can be gamed inside the rules.

Every input is self-reported, so an applicant can claim any figures the joint rules
allow. No rule can close that, so the README states the measured size and what a lender
must do, and the response warns on the crude version, each joint ratio included.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from creditlens.api import app as api
from creditlens.data.domain import INTEGER_FIELDS
from creditlens.data.generator import DatasetConfig, generate_credit_dataset, train_test_split
from creditlens.decision import probability_to_score
from creditlens.features.engineering import MODEL_INPUTS

ROOT = Path(__file__).resolve().parents[1]
APPLICANT = {
    "annual_income": 75000, "loan_amount": 12000, "loan_term_months": 36, "debt_to_income": 0.32,
    "employment_years": 4, "fico_score": 720, "credit_utilization": 0.25, "num_delinquencies": 0,
    "months_since_delinquency": -1, "num_hard_inquiries": 1, "credit_age_months": 84,
    "monthly_inflow": 6250, "monthly_outflow": 4800, "paycheck_regularity": 0.92,
    "bill_payment_consistency": 0.88, "nsf_count": 0, "overdraft_days": 0, "rent_payment_score": 0.95,
    "utility_payment_score": 0.85, "banking_tenure_months": 48, "avg_account_balance": 8500, "savings_rate": 0.1,
}
PATTERN = "in the most favourable tenth of the training data at once"


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        api._startup_train()
        yield c


def _warnings(client, **changes) -> list[str]:
    body = {**APPLICANT, **changes}
    responses = (client.post("/score", json=body).json()["warnings"],
                 client.post("/explain/0", json=body).json()["warnings"],
                 client.post("/score/batch", json={"applicants": [body]}).json()["results"][0]["warnings"])
    assert responses[0] == responses[1] == responses[2]
    return responses[0]


@pytest.mark.parametrize("ratio, side", [(1.22, "above 1.198"), (0.79, "below 0.802")])
def test_the_inflow_ratio_is_flagged_like_the_others(client, ratio, side):
    """Inflow at 1.22 x (training maximum 1.1999) scored 995 with warnings: []."""
    inflow = 6250 * ratio
    warnings = _warnings(client, monthly_inflow=inflow, monthly_outflow=inflow * 0.75)
    assert [w for w in warnings if w.startswith("monthly_inflow / (annual_income / 12)")] == [
        f"monthly_inflow / (annual_income / 12) is {ratio:.3f}, {side}, outside the central 99% of the "
        "training data: treat the score as less certain"]


def test_an_inflow_ratio_inside_the_band_is_not_flagged(client):
    assert _warnings(client, monthly_inflow=6250 * 1.19, monthly_outflow=6250 * 1.19 * 0.75) == []


@pytest.mark.parametrize("changes, named", [
    ({"monthly_inflow": 7250, "monthly_outflow": 3900}, "inflow, outflow are each"),
    ({"monthly_inflow": 7250, "monthly_outflow": 5500, "avg_account_balance": 17500}, "inflow, balance are each"),
    ({"monthly_outflow": 3400, "avg_account_balance": 17500}, "outflow, balance are each"),
    ({"monthly_inflow": 7250, "monthly_outflow": 3900, "avg_account_balance": 17500},
     "inflow, outflow, balance are each"),
])
def test_two_ratios_at_their_favourable_edge_together_are_flagged(client, changes, named):
    pattern = [w for w in _warnings(client, **changes) if PATTERN in w]
    assert len(pattern) == 1 and pattern[0].startswith(named), pattern
    assert "self-reported" in pattern[0] and "verify them independently" in pattern[0]


@pytest.mark.parametrize("changes", [
    {"monthly_inflow": 7250, "monthly_outflow": 5500},
    {"monthly_outflow": 3400},
    {"avg_account_balance": 17500},
])
def test_one_ratio_at_its_edge_is_not_the_pattern(client, changes):
    assert not [w for w in _warnings(client, **changes) if PATTERN in w]


def test_a_warning_moves_no_score(client):
    plain = client.post("/score", json=APPLICANT).json()
    assert (plain["credit_score"], plain["decision"], plain["warnings"]) == (993, "Approved", [])


def _measure(client, inflow: float, outflow: float, balance: float) -> dict:
    """Claim these ratios for every not-approved held-out applicant, inside each band."""
    df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=api.DEFAULT_SEED))
    _, test = train_test_split(df, test_size=0.20, seed=api.DEFAULT_SEED)
    p = api._model_state["model"].predict_proba(api._model_state["engineer"].transform(test)[0])[:, 1]
    not_approved = test[probability_to_score(p) < 600]
    out = {"not_approved": len(not_approved), "flips": 0, "defaulted": 0, "unwarned": 0, "unwarned_defaulted": 0,
           "defaulters": []}
    for _, row in not_approved.iterrows():
        monthly = row["annual_income"] / 12
        honest = {k: (int(row[k]) if k in INTEGER_FIELDS else float(row[k])) for k in MODEL_INPUTS}
        stated_inflow = min(inflow * monthly, 38000.0)
        claim = {**honest, "monthly_inflow": stated_inflow, "monthly_outflow": max(stated_inflow * outflow, 640.0),
                 "avg_account_balance": min(balance * monthly, 81000.0)}
        before = client.post("/score", json=honest).json()
        after = client.post("/score", json=claim)
        if after.status_code != 200 or after.json()["decision"] != "Approved":
            continue
        after = after.json()
        out["flips"] += 1
        out["defaulted"] += int(row["default"])
        unwarned = not [w for w in after["warnings"] if w not in before["warnings"]]
        out["unwarned"] += unwarned
        out["unwarned_defaulted"] += unwarned and int(row["default"])
        if row["default"]:
            out["defaulters"].append((before["credit_score"], before["decision"], after["credit_score"]))
    return out


def test_the_readme_states_the_attack_and_what_it_leaves_open(client):
    text = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    attack = _measure(client, 1.2, 0.503, 2.988)
    assert (attack["not_approved"], attack["flips"], attack["defaulted"]) == (225, 56, 25)
    assert (f"moves {attack['flips']} of the {attack['not_approved']} not-approved applicants among the 2,000 held "
            f"out to Approved; {attack['defaulted']} of those {attack['flips']} defaulted") in text
    assert min(attack["defaulters"]) == (300, "Denied", 616)
    assert "one defaulter goes from 300 / Denied to 616 / Approved" in text
    assert attack["unwarned"] == 0  # the pattern warning catches every one of them
    assert f"so all {attack['flips']} carry a new warning" in text

    inside = _measure(client, 1.14, 0.56, 2.73)
    assert (f"still moves {inside['flips']} to Approved ({inside['defaulted']} defaulted), {inside['unwarned']} of "
            f"them with no new warning ({inside['unwarned_defaulted']} defaulted)") in text
    assert inside["unwarned"] > 0  # the warning does not close this, and the README says so


def test_honest_applicants_rarely_draw_the_pattern_warning(client):
    df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=api.DEFAULT_SEED))
    _, test = train_test_split(df, test_size=0.20, seed=api.DEFAULT_SEED)
    rows = [{k: (int(r[k]) if k in INTEGER_FIELDS else float(r[k])) for k in MODEL_INPUTS}
            for _, r in test.iterrows()]
    results = client.post("/score/batch", json={"applicants": rows[:1000]}).json()["results"]
    results += client.post("/score/batch", json={"applicants": rows[1000:]}).json()["results"]
    flagged = sum(any(PATTERN in w for w in r["warnings"]) for r in results)
    text = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    assert f"{flagged} of the 2,000 unmodified held-out applicants ({flagged / 20:.1f}%) carry it too" in text


def test_the_lender_is_told_to_verify():
    for name in ("README.md", "SECURITY.md"):
        text = re.sub(r"\s+", " ", (ROOT / name).read_text(encoding="utf-8"))
        assert "self-reported" in text and "verify cash flow independently" in text, name
