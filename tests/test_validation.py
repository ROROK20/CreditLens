"""Input validation on the public API.

Bad input is a 4xx, never a 500 and never a silent score: POST /train seed=-1, an
infinite income, a loan term of 0 or -36, months_since_delinquency -999999, an oversized
body, and an unknown or misspelled field. Each case is pinned here, and SECURITY.md's
bounds table is checked against the live request schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from creditlens.api.app import (
    MAX_BODY_BYTES, ApplicantFeatures, BatchScoreRequest, TrainRequest, app,
)

ROOT = Path(__file__).resolve().parents[1]

BASE = {
    "annual_income": 75000, "loan_amount": 12000, "fico_score": 680,
    "debt_to_income": 0.32, "employment_years": 4, "credit_utilization": 0.45,
    "credit_age_months": 84, "monthly_inflow": 6250, "monthly_outflow": 4800,
    "avg_account_balance": 8500, "num_hard_inquiries": 2,
    # Every model input is required, so these are supplied explicitly.
    "loan_term_months": 36, "num_delinquencies": 0, "months_since_delinquency": -1,
    "paycheck_regularity": 0.8, "bill_payment_consistency": 0.8, "nsf_count": 0,
    "overdraft_days": 0, "rent_payment_score": 0.8, "utility_payment_score": 0.8,
    "banking_tenure_months": 12, "savings_rate": 0.1,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _raw(client, path: str, body: str):
    return client.post(path, content=body, headers={"content-type": "application/json"})


# --- malformed input ----------------------------------------------------------

@pytest.mark.parametrize("seed", [-1, 2**32])
def test_train_rejects_an_out_of_range_seed(client, seed):
    assert client.post("/train", json={"seed": seed}).status_code == 422


@pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "NaN"])
@pytest.mark.parametrize("field", ["annual_income", "monthly_inflow", "debt_to_income", "monthly_net_cashflow"])
def test_score_rejects_non_finite_numbers(client, field, literal):
    # Python's JSON parser accepts these literals; the schema must not.
    body = json.dumps({**BASE, field: 1.0}).replace(f'"{field}": 1.0', f'"{field}": {literal}')
    res = _raw(client, "/score", body)
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"][-1] == field


@pytest.mark.parametrize("term", [0, -36, 481])
def test_score_rejects_an_impossible_loan_term(client, term):
    assert client.post("/score", json={**BASE, "loan_term_months": term}).status_code == 422


@pytest.mark.parametrize("months", [-999999, -2, 121, 1201])
def test_score_rejects_an_impossible_delinquency_recency(client, months):
    payload = {**BASE, "num_delinquencies": 1, "months_since_delinquency": months}
    assert client.post("/score", json=payload).status_code == 422


def test_delinquency_recency_must_agree_with_the_count(client):
    """-1 means "never delinquent", exactly as in the training data."""
    contradictions = [
        {"num_delinquencies": 0, "months_since_delinquency": 12},  # recency, no delinquency
        {"num_delinquencies": 2},                                    # delinquent, "never"
    ]
    for extra in contradictions:
        res = client.post("/score", json={**BASE, **extra})
        assert res.status_code == 422, extra
        assert "months_since_delinquency" in res.text
    ok = client.post("/score", json={**BASE, "num_delinquencies": 2, "months_since_delinquency": 12})
    assert ok.status_code == 200


@pytest.mark.parametrize("field, value", [
    ("loan_purpose", "auto"), ("num_open_accounts", 7), ("num_subscriptions", 3), ("interest_rate", 0.1),
])
def test_fields_with_no_effect_on_default_are_not_inputs(client, field, value):
    """These have no effect on the synthetic default label (data/generator.py), so they
    are not inputs."""
    res = client.post("/score", json={**BASE, field: value})
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"][-1] == field


# A JSON "\ud800" escape parses to a Python string UTF-8 cannot encode. The 422's
# own echo of it raised UnicodeEncodeError, so every POST answered 500 to
# {"\ud800": 1}, on an API whose security policy says 422 for anything invalid.
SURROGATE_BODIES = {
    "as a key": '{"\\ud800": 1}',
    "as a nested key": '{"\\ud800": {"\\ud800": ["\\ud800"]}}',
    "as a value": '{"fico_score": "\\ud800"}',
    "inside a long value": '{"fico_score": "' + "x" * 200 + '\\ud800"}',
    "in a list element": '{"applicants": [{"\\udfff": 1}]}',
    "as a valid-shaped train field": '{"seed": "\\ud800"}',
}


@pytest.mark.parametrize("path", ["/score", "/score/batch", "/explain/0", "/train"])
@pytest.mark.parametrize("body", SURROGATE_BODIES.values(), ids=SURROGATE_BODIES.keys())
def test_a_lone_surrogate_is_a_422_on_every_post(client, path, body):
    res = _raw(client, path, body)
    assert res.status_code == 422, (path, body, res.status_code)
    assert res.json()["detail"], res.text[:200]


def test_a_lone_surrogate_in_a_text_field_is_refused_not_a_500(client):
    body = json.dumps({**BASE, "demographic_group": "\ud800"})  # json.dumps escapes it
    for path in ("/score", "/explain/0"):
        res = _raw(client, path, body)
        assert res.status_code == 422, (path, res.status_code)
        assert res.json()["detail"][0]["loc"][-1] == "demographic_group"


@pytest.mark.parametrize("body", ['{"a": ' + "9" * 5000 + "}", "[" * 100_000],
                         ids=["a 5,000-digit integer", "100,000 open brackets"])
def test_a_body_python_cannot_parse_is_a_400_as_documented(client, body):
    assert _raw(client, "/score", body).status_code == 400
    assert "`400`" in (ROOT / "SECURITY.md").read_text(encoding="utf-8")


def test_a_huge_unknown_field_is_not_echoed_back(client):
    big = client.post("/score", json={**BASE, "loan_purpose": "x" * 100_000})
    assert big.status_code == 422
    assert len(big.content) < 2000  # the 422 does not echo 100 kB back


def test_a_long_refusal_reason_reaches_the_client_whole(client):
    """The cap meant for the echoed input also cut the API's own reason, and the live
    page showed it: "must be at most 2.8 x monthly... (142 characters)"."""
    res = client.post("/score", json={**BASE, "annual_income": 15000, "monthly_inflow": 1250, "monthly_outflow": 960,
                                      "avg_account_balance": 1700, "loan_amount": 49000, "loan_term_months": 12})
    assert res.status_code == 422
    assert res.json()["detail"][0]["msg"] == (
        "Value error, the monthly loan payment (loan_amount / loan_term_months) must be at most 2.8 x monthly "
        "income (the training range); it is 3.27 x")


def test_what_a_422_echoes_stays_capped(client):
    """The reason is whole; the caller's own strings are still bounded, keys included."""
    res = client.post("/score", json={**BASE, "k" * 5000: 1, "demographic_group": "g" * 5000})
    assert res.status_code == 422
    assert len(res.content) < 5000
    assert any("(5000 characters)" in str(e["loc"]) for e in res.json()["detail"])
    assert any("(5000 characters)" in str(e["input"]) for e in res.json()["detail"])



def test_unknown_fields_are_rejected_not_ignored(client):
    # A dropped typo would score this applicant as having 0 delinquencies.
    assert client.post("/score", json={**BASE, "num_delinquency": 3}).status_code == 422
    assert client.post("/train", json={"n_sample": 500}).status_code == 422
    batch = {"applicants": [BASE], "pad": "x"}
    assert client.post("/score/batch", json=batch).status_code == 422


def test_demographic_group_is_bounded(client):
    assert client.post("/score", json={**BASE, "demographic_group": "g" * 65}).status_code == 422


# --- request-size cap ---------------------------------------------------------

@pytest.mark.parametrize("path", ["/score", "/score/batch", "/train", "/explain/0"])
def test_oversized_body_is_rejected_with_413_before_parsing(client, path):
    body = json.dumps({**BASE, "pad": "x" * (MAX_BODY_BYTES + 1)})
    res = _raw(client, path, body)
    assert res.status_code == 413
    assert str(MAX_BODY_BYTES) in res.json()["detail"]


def test_413_carries_cors_headers_so_a_browser_can_read_it(client):
    body = json.dumps({**BASE, "pad": "x" * (MAX_BODY_BYTES + 1)})
    res = client.post("/score", content=body, headers={
        "content-type": "application/json", "origin": "http://localhost:3003"})
    assert res.status_code == 413
    assert res.headers.get("access-control-allow-origin") == "*"


def test_oversized_chunked_body_without_content_length_is_rejected(client):
    def chunks():
        payload = json.dumps({**BASE, "pad": "x" * (MAX_BODY_BYTES + 1)}).encode()
        for i in range(0, len(payload), 65536):
            yield payload[i:i + 65536]

    res = client.post("/score", content=chunks(), headers={"content-type": "application/json"})
    assert res.status_code == 413


def test_the_largest_valid_batch_fits_under_the_cap(client):
    """The cap must never reject a valid request: 1,000 applicants (the batch
    maximum), every field present, every float at full 17-digit precision.
    Every value lies in the validity domain."""
    worst = {name: 0.32345678901234568 for name in ApplicantFeatures.model_fields}
    worst.update({
        "annual_income": 123456.78901234568, "loan_amount": 12345.678901234567,
        "loan_term_months": 60, "fico_score": 712,
        "num_delinquencies": 5, "months_since_delinquency": 119, "num_hard_inquiries": 8,
        "credit_age_months": 123.45678901234568, "monthly_inflow": 12345.678901234567,
        "monthly_outflow": 9876.5432109876543,
        "nsf_count": 6, "overdraft_days": 10, "banking_tenure_months": 123.45678901234568,
        "avg_account_balance": 12345.678901234567,
        "employment_years": 12.345678901234567, "demographic_group": "g" * 64,
    })
    body = json.dumps({"applicants": [worst] * 1000})
    size = len(body.encode())
    assert size < MAX_BODY_BYTES
    assert f"is {size:,} bytes (measured" in re.sub(r"\s+", " ", (ROOT / "SECURITY.md").read_text(encoding="utf-8"))
    res = _raw(client, "/score/batch", body)
    assert res.status_code == 200, res.text[:300]
    assert res.json()["count"] == 1000


# --- SECURITY.md states the schema exactly ------------------------------------

def _num(x) -> str:
    return f"{int(x):,}" if float(x).is_integer() else f"{x:g}"


def _schema_bounds(model) -> dict[str, str]:
    """Human-readable bound for every field of a request model, from its live
    JSON schema — the notation SECURITY.md's tables use."""
    out = {}
    schema = model.model_json_schema()
    for name, prop in schema["properties"].items():
        spec = next((s for s in prop.get("anyOf", []) if s.get("type") != "null"), prop)
        if "enum" in spec:
            out[name] = "one of " + ", ".join(f"`{v}`" for v in spec["enum"])
        elif spec.get("type") == "string":
            out[name] = f"text, at most {spec['maxLength']} characters"
        else:
            low = (f"> {_num(spec['exclusiveMinimum'])}" if "exclusiveMinimum" in spec
                   else _num(spec["minimum"]))
            out[name] = f"{low} to {_num(spec['maximum'])}"
        if name not in schema.get("required", []):
            out[name] += ", or omitted"
    return out


@pytest.mark.parametrize("model", [ApplicantFeatures, TrainRequest, BatchScoreRequest],
                         ids=lambda m: m.__name__)
def test_security_md_bounds_table_matches_the_schema_exactly(model):
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    rows = dict(re.findall(r"^\| `(\w+)` \| (.+?) \|$", text, re.M))
    if model is BatchScoreRequest:  # one list field: its length bounds
        spec = model.model_json_schema()["properties"]["applicants"]
        assert rows["applicants"] == f"{spec['minItems']:,} to {spec['maxItems']:,} applicants"
        return
    documented = {k: v for k, v in rows.items() if k in model.model_fields}
    assert documented == _schema_bounds(model)


# --- contradictory cash flows, coerced types ----------------------------------

# Net cash flow is derived, always, so a redundant field cannot change the score or the
# decision, and /explain sums to the served probability.
REDUNDANT_FIELD_APPLICANT = {
    "annual_income": 75000, "loan_amount": 12000, "loan_term_months": 36, "fico_score": 564,
    "debt_to_income": 0.37, "employment_years": 4, "credit_utilization": 0.6, "credit_age_months": 84,
    "num_hard_inquiries": 2, "num_delinquencies": 0, "months_since_delinquency": -1,
    "monthly_inflow": 6250, "monthly_outflow": 5190.33, "avg_account_balance": 8500,
    "paycheck_regularity": 0.8, "bill_payment_consistency": 0.8, "nsf_count": 0, "overdraft_days": 0,
    "rent_payment_score": 0.8, "utility_payment_score": 0.8, "banking_tenure_months": 12, "savings_rate": 0.1,
}


@pytest.mark.parametrize("path", ["/score", "/explain/0"])
@pytest.mark.parametrize("net", [1059.67, 1059.665, 1234.5])
def test_net_cash_flow_is_not_an_input(client, path, net):
    """Agreeing to the cent or not, sending it is a 422 like any unknown field."""
    res = client.post(path, json={**REDUNDANT_FIELD_APPLICANT, "monthly_net_cashflow": net})
    assert res.status_code == 422, res.text[:200]
    assert res.json()["detail"][0]["loc"][-1] == "monthly_net_cashflow"
    batch = {"applicants": [{**REDUNDANT_FIELD_APPLICANT, "monthly_net_cashflow": net}]}
    assert client.post("/score/batch", json=batch).status_code == 422


def test_the_attribution_sums_to_the_served_probability(client):
    scored = client.post("/score", json=REDUNDANT_FIELD_APPLICANT).json()
    explained = client.post("/explain/0", json=REDUNDANT_FIELD_APPLICANT).json()
    assert explained["credit_score"] == scored["credit_score"]
    assert explained["base_value"] + sum(explained["shap_values"].values()) == pytest.approx(
        scored["default_probability"], abs=5.1e-5)


def test_a_low_derived_net_cash_flow_is_warned_about_on_every_endpoint(client):
    """The warning follows the derived value: an applicant with $70 a month to spare is
    warned, with no field to send."""
    tight = {**BASE, "annual_income": 30000, "loan_amount": 5000, "monthly_inflow": 2500,
             "monthly_outflow": 2430, "avg_account_balance": 3000}
    expected = "monthly_net_cashflow 70 is below 97"
    for path in ("/score", "/explain/0"):
        warnings = client.post(path, json=tight).json()["warnings"]
        assert any(w.startswith(expected) for w in warnings), (path, warnings)
    batch = client.post("/score/batch", json={"applicants": [tight]}).json()["results"][0]["warnings"]
    assert any(w.startswith(expected) for w in batch), batch


@pytest.mark.parametrize("field, value", [
    ("fico_score", "700"), ("nsf_count", True), ("fico_score", 700.0),
    ("annual_income", "75000"), ("credit_utilization", True), ("num_delinquencies", "0"),
])
def test_types_are_strict(client, field, value):
    res = client.post("/score", json={**BASE, field: value})
    assert res.status_code == 422, (field, value)
    assert res.json()["detail"][0]["loc"][-1] == field


def test_a_json_integer_is_still_a_valid_float(client):
    assert client.post("/score", json={**BASE, "annual_income": 75000, "credit_utilization": 0}).status_code == 200


# --- the validity domain -------------------------------------------------------
# Outside the training data the model has no evidence, so these are refused, not scored.

@pytest.mark.parametrize("field, value", [
    ("loan_amount", 10_000_000),
    ("annual_income", 1),
    ("num_delinquencies", 100),
    ("nsf_count", 1000),
    ("fico_score", 300),
])
def test_values_beyond_the_training_data_are_rejected(client, field, value):
    payload = {**BASE, field: value}
    if field == "num_delinquencies":
        payload["months_since_delinquency"] = 6
    res = client.post("/score", json=payload)
    assert res.status_code == 422, (field, value)
    assert field in {e["loc"][-1] for e in res.json()["detail"]}


def test_the_domain_constants_are_the_training_datas_range_plus_margin():
    """The literals in data/domain.py are what the derivation returns for the
    default training data, and that data lies inside them."""
    from creditlens.data import domain
    from creditlens.data.generator import DatasetConfig, generate_credit_dataset

    df = generate_credit_dataset(DatasetConfig())
    derived = domain.derive_validity_domain(df)
    assert derived["domain"] == domain.VALIDITY_DOMAIN
    assert derived["warning_band"] == domain.WARNING_BAND
    assert derived["inflow_to_monthly_income"] == domain.INFLOW_TO_MONTHLY_INCOME
    assert derived["inflow_to_monthly_income_warning"] == domain.INFLOW_TO_MONTHLY_INCOME_WARNING
    assert derived["cash_flow_edge"] == domain.CASH_FLOW_EDGE
    assert derived["outflow_to_inflow"] == domain.OUTFLOW_TO_INFLOW
    assert derived["outflow_to_inflow_warning"] == domain.OUTFLOW_TO_INFLOW_WARNING
    assert derived["balance_to_monthly_income"] == domain.BALANCE_TO_MONTHLY_INCOME
    assert derived["balance_to_monthly_income_warning"] == domain.BALANCE_TO_MONTHLY_INCOME_WARNING
    assert derived["max_payment_burden"] == domain.MAX_PAYMENT_BURDEN
    assert derived["payment_burden_warning"] == domain.PAYMENT_BURDEN_WARNING
    for name, (lo, hi) in domain.VALIDITY_DOMAIN.items():
        observed = df[name][df[name] >= 0] if name == "months_since_delinquency" else df[name]
        assert lo <= observed.min() and observed.max() <= hi, name
    # Every training applicant is accepted by the API.
    numeric = [k for k in ApplicantFeatures.model_fields if k != "demographic_group"]
    for row in df.sample(300, random_state=0).to_dict("records"):
        ApplicantFeatures(**{
            k: int(row[k]) if ApplicantFeatures.model_fields[k].annotation is int else float(row[k])
            for k in numeric})


def test_inflow_must_match_income(client):
    """A higher stated income with the same deposits is not an applicant the
    model has seen: inflow must be 0.78-1.22 x annual_income / 12."""
    monthly = BASE["annual_income"] / 12
    for ratio, ok in ((0.78, True), (1.22, True), (0.77, False), (1.23, False), (0.5, False)):
        res = client.post("/score", json={**BASE, "monthly_inflow": monthly * ratio,
                                          "monthly_outflow": monthly * ratio * 0.7})
        assert (res.status_code == 200) is ok, (ratio, res.text[:200])
        if not ok:
            assert "monthly_inflow must be 0.78-1.22 x annual_income / 12" in res.text
    doubled = {**BASE, "annual_income": BASE["annual_income"] * 2}
    assert client.post("/score", json=doubled).status_code == 422


def test_the_monthly_payment_is_bounded_by_income(client):
    # $49,000 over 9 months on $12,000 a year: 5.4 x monthly income.
    res = client.post("/score", json={**BASE, "annual_income": 12000, "monthly_inflow": 1000,
                                      "monthly_outflow": 800, "avg_account_balance": 1500,
                                      "loan_amount": 49000, "loan_term_months": 9})
    assert res.status_code == 422
    assert "monthly loan payment" in res.text


def test_values_near_the_edge_are_scored_and_flagged(client):
    edge = {**BASE, "fico_score": 400, "nsf_count": 5}
    for path in ("/score", "/explain/0"):
        body = client.post(path, json=edge).json()
        assert body["warnings"] and len(body["warnings"]) == 2, body.get("warnings")
        assert body["warnings"][0].startswith("fico_score 400 is below 475")
        assert body["warnings"][1].startswith("nsf_count 5 is above 3")
    batch = client.post("/score/batch", json={"applicants": [BASE, edge]}).json()["results"]
    assert [len(r["warnings"]) for r in batch] == [0, 2]
    assert client.post("/score", json=BASE).json()["warnings"] == []


class TestComputedNetCashflowIsInsideTheDomain:
    """Net cash flow is derived, and the derived value answers to its bound: $57,000 a
    month out on $6,250 in is refused."""

    APPLICANT = {"annual_income": 75000, "loan_amount": 12000, "loan_term_months": 36,
                 "debt_to_income": 0.32, "employment_years": 4, "fico_score": 720,
                 "credit_utilization": 0.25, "num_delinquencies": 0,
                 "months_since_delinquency": -1, "num_hard_inquiries": 1, "credit_age_months": 84,
                 "monthly_inflow": 6250, "monthly_outflow": 4800, "paycheck_regularity": 0.92,
                 "bill_payment_consistency": 0.88, "nsf_count": 0, "overdraft_days": 0,
                 "rent_payment_score": 0.95, "utility_payment_score": 0.85,
                 "banking_tenure_months": 48, "avg_account_balance": 8500, "savings_rate": 0.1}

    @pytest.mark.parametrize("flows", [
        {"monthly_outflow": 57000},  # net -50,750
        {"monthly_outflow": 7451},  # net -1,201: one dollar past the floor
        {"annual_income": 600000, "monthly_inflow": 61000, "monthly_outflow": 1000},  # net 60,000
        # Inside the outflow-to-inflow rule, past net's ceiling.
        {"annual_income": 600000, "monthly_inflow": 61000, "monthly_outflow": 30000,
         "avg_account_balance": 60000},  # net 31,000
    ])
    def test_outside_the_range_is_refused(self, client, flows):
        res = client.post("/score", json={**self.APPLICANT, **flows})
        assert res.status_code == 422, res.text
        assert "monthly_inflow - monthly_outflow" in res.text

    # 7450 would be net -1,200, in range, but spending 1.19 x inflow, which no training
    # applicant does, so it is a 422 (see the next class).
    @pytest.mark.parametrize("outflow", [6100, 4800])
    def test_inside_the_range_is_scored(self, client, outflow):
        assert client.post("/score", json={**self.APPLICANT, "monthly_outflow": outflow}).status_code == 200

    def test_every_training_row_is_inside_the_range(self):
        from creditlens.data.domain import VALIDITY_DOMAIN
        from creditlens.data.generator import generate_credit_dataset

        df = generate_credit_dataset()
        net = df["monthly_inflow"] - df["monthly_outflow"]
        lo, hi = VALIDITY_DOMAIN["monthly_net_cashflow"]
        assert net.min() >= lo and net.max() <= hi


class TestTheGeneratorsOtherCouplings:
    """The generator draws outflow as inflow x U(0.50, 0.95) and the balance as
    income / 12 x U(0.5, 3.0), and the API holds a request to both."""

    APPLICANT = TestComputedNetCashflowIsInsideTheDomain.APPLICANT

    @pytest.mark.parametrize("outflow, ok", [(2800, False), (2950, True), (6100, True), (6140, False)])
    def test_outflow_must_match_inflow(self, client, outflow, ok):
        res = client.post("/score", json={**self.APPLICANT, "monthly_outflow": outflow})
        assert (res.status_code == 200) is ok, (outflow, res.text[:200])
        if not ok:
            assert "monthly_outflow must be" in res.text and "x monthly_inflow" in res.text

    @pytest.mark.parametrize("balance, ok", [(2200, False), (2400, True), (19000, True), (21000, False)])
    def test_balance_must_match_income(self, client, balance, ok):
        res = client.post("/score", json={**self.APPLICANT, "avg_account_balance": balance})
        assert (res.status_code == 200) is ok, (balance, res.text[:200])
        if not ok:
            assert "avg_account_balance must be" in res.text and "x annual_income / 12" in res.text

    def test_claiming_both_can_no_longer_flip_a_not_approved_applicant(self, client):
        """A moderately weak applicant, then claiming almost no spending and a balance
        far above anything income supports."""
        weak = {**self.APPLICANT, "fico_score": 610, "debt_to_income": 0.5, "credit_utilization": 0.7,
                "nsf_count": 2}
        assert client.post("/score", json=weak).status_code == 200
        claim = {**weak, "monthly_outflow": 410, "avg_account_balance": 170000}
        assert client.post("/score", json=claim).status_code == 422

    def test_near_the_edge_is_scored_and_flagged(self, client):
        near = {**self.APPLICANT, "monthly_outflow": 6050, "avg_account_balance": 18800}
        warnings = client.post("/score", json=near).json()["warnings"]
        assert any(w.startswith("monthly_outflow / monthly_inflow is 0.968") for w in warnings), warnings
        assert any(w.startswith("avg_account_balance / (annual_income / 12) is 3.008") for w in warnings), warnings

    def test_training_data_is_inside_both_rules(self):
        from creditlens.data.domain import BALANCE_TO_MONTHLY_INCOME, OUTFLOW_TO_INFLOW
        from creditlens.data.generator import generate_credit_dataset

        df = generate_credit_dataset()
        outflow = df["monthly_outflow"] / df["monthly_inflow"]
        balance = df["avg_account_balance"] / (df["annual_income"] / 12)
        assert OUTFLOW_TO_INFLOW[0] <= outflow.min() and outflow.max() <= OUTFLOW_TO_INFLOW[1]
        assert BALANCE_TO_MONTHLY_INCOME[0] <= balance.min() and balance.max() <= BALANCE_TO_MONTHLY_INCOME[1]
