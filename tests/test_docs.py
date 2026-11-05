"""Documentation truth tests: every number the docs show must be what the code
prints today, reproduced by the steps the docs themselves give (a number and
its date are one claim, and both go stale silently).

These run the documented examples in-process (FastAPI TestClient — no ports)
and compare against the text in README.md and docs/usage.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from creditlens.api.app import _startup_train, app

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "docs" / "usage.md"]

# POST examples: `curl -X POST .../<endpoint> \ -H ... \ -d '<json>'`
_CURL_POST = re.compile(
    r"curl -X POST http://localhost:8002/(score/batch|score|train|explain/\d+) \\\n"
    r"\s+-H \"Content-Type: application/json\" \\\n"
    r"\s+-d '(\{.*?\})'\n",
    re.S,
)
# GET examples: `curl http://localhost:8002/<path>` on a line of its own.
_CURL_GET = re.compile(r"^curl http://localhost:8002/([a-z/]+)\n", re.M)


def _documented_response(text: str, start: int) -> str | None:
    """The JSON object documented in the `# ...` comment lines right after an
    example, or None when the example documents no response."""
    documented: list[str] = []
    for line in text[start:].splitlines():
        if not line.startswith("#"):
            break
        body = line[1:].strip()
        if documented or body.startswith("{"):
            documented.append(body)
            joined = " ".join(documented)
            if joined.count("{") == joined.count("}"):
                return joined  # the documented JSON object is complete
    return None


def _documented_examples(path: Path) -> list[tuple[str, dict | None, str]]:
    """(endpoint, payload-or-None-for-GET, documented-response-text) for each
    curl example followed by a documented JSON response."""
    text = path.read_text(encoding="utf-8")
    examples = []
    for m in _CURL_POST.finditer(text):
        documented = _documented_response(text, m.end())
        if documented:
            examples.append((m.group(1), json.loads(m.group(2)), documented))
    for m in _CURL_GET.finditer(text):
        documented = _documented_response(text, m.end())
        if documented:
            examples.append((m.group(1), None, documented))
    return examples


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_docs_contain_score_examples():
    # Non-vacuity: the parser must actually find the documented examples.
    found = {p.name: [e[0] for e in _documented_examples(p)] for p in DOCS}
    assert "score" in found["README.md"], "README /score example not found"
    for endpoint in ("score", "score/batch", "train", "explain/1", "health", "model/metrics", "fairness/report"):
        assert endpoint in found["usage.md"], f"usage.md {endpoint} example not found"


def _check(client, path: Path, step: str, endpoint: str, payload: dict | None, documented: str):
    where = f"{path.name} {endpoint} ({step})"
    if payload is None:
        live = client.get(f"/{endpoint}").json()
        # GET examples may be abridged: every documented key must match.
        shown = json.loads(documented)
        assert {k: live.get(k) for k in shown} == shown, where
        return
    live = client.post(f"/{endpoint}", json=payload).json()
    if endpoint == "score/batch":
        shown = [int(s) for s in re.findall(r'"credit_score": (\d+)', documented)]
        assert shown == [r["credit_score"] for r in live["results"]], where
    elif endpoint.startswith("explain/"):
        # Abridged with "...": every value it does show must match.
        assert int(re.search(r'"credit_score": (\d+)', documented).group(1)) == live["credit_score"], where
        assert re.search(r'"decision": "(\w+)"', documented).group(1) == live["decision"], where
        assert float(re.search(r'"base_value": ([\d.]+)', documented).group(1)) == round(live["base_value"], 4), where
        assert "attribution_players" in live, where
    else:
        assert json.loads(documented) == live, where


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_documented_scores_reproduce_at_startup_and_after_post_train(client, path):
    """Every documented response reproduces from the startup model AND after
    the documented `POST /train` (body-less) step."""
    for step in ("startup", "after POST /train"):
        if step == "startup":
            _startup_train()
        else:
            assert client.post("/train").status_code == 200
        for endpoint, payload, documented in _documented_examples(path):
            _check(client, path, step, endpoint, payload, documented)


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_every_documented_post_example_is_accepted(client, path):
    """An example the API refuses teaches a 422, so every documented POST is accepted."""
    text = path.read_text(encoding="utf-8")
    found = [(m.group(1), json.loads(m.group(2))) for m in _CURL_POST.finditer(text)]
    assert found
    for endpoint, payload in found:
        res = client.post(f"/{endpoint}", json=payload)
        assert res.status_code == 200, (path.name, endpoint, res.text[:300])
    _startup_train()


def test_readme_quick_start_prints_exactly_what_it_documents():
    """Run the documented block and require its documented output, byte for byte."""
    import contextlib
    import io

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## Quick Start\n"):readme.index("## Web Demo / Frontend")]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    documented = re.search(r"```text\n(.*?)```", section, re.S).group(1)

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "README.md#quick-start", "exec"), {"__name__": "quick_start"})
    assert out.getvalue() == documented


def test_model_behaviour_numbers_reproduce(client):
    """README "Model behaviour" numbers are measured, not remembered."""
    import numpy as np

    from creditlens.api import app as api
    from creditlens.data.generator import DatasetConfig, generate_credit_dataset, train_test_split
    from creditlens.decision import APPROVE_MIN_SCORE, probability_to_score

    _startup_train()
    df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=api.DEFAULT_SEED))
    train_df, test_df = train_test_split(df, test_size=0.20, seed=api.DEFAULT_SEED)
    model, engineer = api._model_state["model"], api._model_state["engineer"]
    X, _ = engineer.transform(test_df)
    p = model.predict_proba(X)[:, 1]
    scores = probability_to_score(p)
    approved = int((scores >= APPROVE_MIN_SCORE).sum())
    corr = np.corrcoef(train_df["fico_score"], train_df["interest_rate"])[0, 1]

    n = len(test_df)
    observed = test_df["default"].to_numpy()
    fifth = pd.qcut(p, 5, labels=False)
    predicted_q = " / ".join(f"{p[fifth == i].mean():.1%}" for i in range(5))
    observed_q = " / ".join(f"{observed[fifth == i].mean():.1%}" for i in range(5))

    delinquent = test_df[test_df["num_delinquencies"] > 0]
    recency = [probability_to_score(model.predict_proba(
        engineer.transform(delinquent.assign(months_since_delinquency=m))[0])[:, 1]).mean()
        for m in (24, 60, 119)]
    assert max(recency) - min(recency) == 0

    fields = [f for f in api.ApplicantFeatures.model_fields if f != "demographic_group"]
    denied = test_df.index[scores < APPROVE_MIN_SCORE][:60]
    assert len(denied) == 60
    fico_first = sum(
        client.post("/explain/0", json={k: (int(test_df.at[i, k]) if api.ApplicantFeatures.model_fields[k].annotation is int
                                            else float(test_df.at[i, k])) for k in fields})
        .json()["top_negative_factors"][0]["feature"] == "fico_score"
        for i in denied)

    live = client.get("/fairness/report").json()["checks"]

    def interval(name: str) -> str:
        c = live[name]
        return f"{c['value']:.3f}, CI {c['ci'][0]:.3f}–{c['ci'][1]:.3f}"

    text = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    for claim in (
        f"P(default) = 0 for {int((p == 0).sum())} of the {n:,}",
        f"P(default) = 1 for {int((p == 1).sum())} of the {n:,}",
        f"P(default) runs from {p.min():.4f} to {p.max():.4f} (scores {scores.min()} to {scores.max()};",
        f"{predicted_q} predicted against {observed_q} observed",
        f"rate on the training data: {corr:.2f})",
        f"about {0.40 / train_df['default'].mean():.1f} times the {train_df['default'].mean():.1%} training default rate",
        f"approves {approved} of the {n:,} audited applicants ({approved / n:.1%})",
        f"top risk factor in {fico_first} of the first 60 not-approved audit applicants",
        f"mean score is flat past it ({recency[0]:.0f} from 24 to 119 months)",
        f"audit applicant ({int((scores < APPROVE_MIN_SCORE).sum())} of the {n:,})",
        f"Demographic parity ({interval('demographic_parity')})",
        f"equal opportunity ({interval('equal_opportunity')})",
        f"disparate impact ({interval('disparate_impact')})",
        f"equalized odds ({interval('equalized_odds')})",
    ):
        assert claim in text, claim


def test_readme_opt_in_bias_sentences_reproduce():
    """The README reads each biased dataset's verdict in prose, outside the
    block the Quick Start test runs."""
    from creditlens.data.generator import DatasetConfig, generate_credit_dataset, train_test_split
    from creditlens.fairness.metrics import FairnessAuditor
    from creditlens.features.engineering import FeatureEngineer
    from creditlens.models.scorer import LogisticScorer

    def audit(shift, seed):
        data = generate_credit_dataset(DatasetConfig(n_samples=10_000, seed=seed, group_c_fico_shift=shift))
        train, test = train_test_split(data)
        engineer = FeatureEngineer()
        X, y = engineer.fit_transform(train)
        X_test, y_test = engineer.transform(test)
        model = LogisticScorer().fit(X, y)
        return FairnessAuditor().audit(y_test.values, model.predict_proba(X_test)[:, 1],
                                       test["demographic_group"].values)

    def verdicts(report):
        return {name: c.verdict for name, c in report.checks.items()}

    at_60, at_80, at_100, seed_7 = audit(60, 42), audit(80, 42), audit(100, 42), audit(80, 7)
    assert [r.overall_verdict for r in (at_60, at_80, at_100, seed_7)] == ["INCONCLUSIVE", "FAIL", "FAIL", "FAIL"]
    assert verdicts(at_60) == {"demographic_parity": "INCONCLUSIVE", "equal_opportunity": "INCONCLUSIVE",
                               "equalized_odds": "INCONCLUSIVE", "disparate_impact": "INCONCLUSIVE"}
    assert verdicts(at_80) == {"demographic_parity": "FAIL", "equal_opportunity": "FAIL",
                               "equalized_odds": "FAIL", "disparate_impact": "INCONCLUSIVE"}

    text = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    eo, dp, di = at_80.checks["equalized_odds"], at_60.checks["demographic_parity"], at_80.checks["disparate_impact"]
    eo60 = at_60.checks["equalized_odds"]
    for claim in (
        f"approval when repaid falls to {at_80.group_metrics['C'].approval_rate_repaid:.1%} "
        f"against {at_80.group_metrics['A'].approval_rate_repaid:.1%} for group A",
        f"interval from {eo.ci_low:.3f}",
        f"equalized odds included ({eo60.value:.3f}, CI {eo60.ci_low:.3f}–{eo60.ci_high:.3f})",
        f"parity there is INCONCLUSIVE ({dp.value:.3f}, CI {dp.ci_low:.3f}–{dp.ci_high:.3f})",
        f"Disparate impact reads {di.value:.3f}, {'above' if di.value >= 0.8 else 'below'} 0.80 on the "
        f"point estimate, but its interval ({di.ci_low:.3f}–{di.ci_high:.3f})",
    ):
        assert claim in text, claim


def test_requirements_header_numbers_are_current(client):
    """requirements.txt quotes the default applicant's score and the disparate
    impact ratio to explain why the pins matter, so they must be the current ones."""
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    m = re.search(r"default applicant -> (\d+), disparate_impact_ratio ([\d.]+)", re.sub(r"\s*#\s*", " ", text))
    assert m, "requirements.txt does not quote the numbers this test checks"
    _startup_train()
    default = {"annual_income": 75000, "loan_amount": 12000, "loan_term_months": 36,
               "debt_to_income": 0.32, "employment_years": 4, "fico_score": 720,
               "credit_utilization": 0.25, "num_delinquencies": 0,
               "months_since_delinquency": -1, "num_hard_inquiries": 1, "credit_age_months": 84,
               "monthly_inflow": 6250, "monthly_outflow": 4800, "paycheck_regularity": 0.92,
               "bill_payment_consistency": 0.88, "nsf_count": 0, "overdraft_days": 0,
               "rent_payment_score": 0.95, "utility_payment_score": 0.85,
               "banking_tenure_months": 48, "avg_account_balance": 8500, "savings_rate": 0.1,}
    assert int(m.group(1)) == client.post("/score", json=default).json()["credit_score"]
    assert float(m.group(2)) == client.get("/fairness/report").json()["disparate_impact_ratio"]


def _frontend_to_api_names() -> dict[str, str]:
    api = (ROOT / "frontend" / "lib" / "api.ts").read_text(encoding="utf-8")
    body = api[api.index("function toApiApplicant"):]
    return {camel: snake for snake, camel in re.findall(r"(\w+): (?:.*\? )?a\.(\w+)", body)}


def test_every_frontend_slider_stays_inside_the_validity_domain():
    """The API answers 422 outside the training data's range, so a slider that could
    leave it would hand the user a refusal instead of a score."""
    from creditlens.data.domain import (
        BALANCE_TO_MONTHLY_INCOME, INFLOW_TO_MONTHLY_INCOME, OUTFLOW_TO_INFLOW, VALIDITY_DOMAIN,
    )

    page = (ROOT / "frontend" / "app" / "page.tsx").read_text(encoding="utf-8")
    names = _frontend_to_api_names()
    sliders = re.findall(r'\{ key: "(\w+)",[^}]*?min: ([\d.]+), max: ([\d.]+)', page)
    assert len(sliders) == 22
    for key, lo, hi in sliders:
        dom_lo, dom_hi = VALIDITY_DOMAIN[names[key]]
        assert dom_lo <= float(lo) and float(hi) <= dom_hi, (key, lo, hi, dom_lo, dom_hi)

    ts = (ROOT / "frontend" / "lib" / "scoring.ts").read_text(encoding="utf-8")
    lo, hi = re.search(r"INFLOW_TO_MONTHLY_INCOME = \[([\d.]+), ([\d.]+)\]", ts).groups()
    assert (float(lo), float(hi)) == INFLOW_TO_MONTHLY_INCOME
    out_lo, out_hi = re.search(r"OUTFLOW_TO_INFLOW = \[([\d.]+), ([\d.]+)\]", ts).groups()
    assert (float(out_lo), float(out_hi)) == OUTFLOW_TO_INFLOW
    bal_lo, bal_hi = re.search(r"BALANCE_TO_MONTHLY_INCOME = \[([\d.]+), ([\d.]+)\]", ts).groups()
    assert (float(bal_lo), float(bal_hi)) == BALANCE_TO_MONTHLY_INCOME
    # Every preset that sets an income keeps its flows and balance inside the rules.
    presets = re.findall(r'preset === "(\w+)"\) setApplicant\(\{(.*?)\}\);', page, re.S)
    assert {p for p, _ in presets} == {"strong", "median", "weak"}
    for name, body in presets:
        income = float(re.search(r"annualIncome: (\d+)", body).group(1))
        inflow = float(re.search(r"monthlyInflow: (\d+)", body).group(1))
        outflow = float(re.search(r"monthlyOutflow: (\d+)", body).group(1))
        balance = float(re.search(r"avgAccountBalance: (\d+)", body).group(1))
        assert float(lo) <= inflow / (income / 12) <= float(hi), name
        assert OUTFLOW_TO_INFLOW[0] <= outflow / inflow <= OUTFLOW_TO_INFLOW[1], name
        assert BALANCE_TO_MONTHLY_INCOME[0] <= balance / (income / 12) <= BALANCE_TO_MONTHLY_INCOME[1], name
    # Moving income carries the balance too, and these sliders follow the rules.
    assert "avgAccountBalance: Math.round(applicant.avgAccountBalance * scale)" in page
    assert "OUTFLOW_TO_INFLOW[0]" in page and "BALANCE_TO_MONTHLY_INCOME[0]" in page


def test_the_documented_test_count_is_the_collected_count(request):
    """A count and its date are one claim, in README, requirements.txt, setup.py and
    CONTRIBUTING.md. Checked only when the whole suite is collected: a partial run cannot
    know the total."""
    option = request.config.option
    files = {p.name for p in (ROOT / "tests").glob("test_*.py")}
    if ({i.path.name for i in request.session.items} != files or option.keyword or option.markexpr
            or getattr(option, "lf", False) or getattr(option, "failedfirst", False) or option.deselect):
        return
    collected = len(request.session.items)
    claims = {
        "README.md": r"# (\d+) passed, 0 skipped +# re-run (20\d\d-\d\d-\d\d)",
        "requirements.txt": r"suite re-run (?P<date>20\d\d-\d\d-\d\d) on this pinned set:\s*#\s*(?P<n>\d+) passed, 0 skipped",
        "setup.py": r"suite re-run (?P<date>20\d\d-\d\d-\d\d) on the same\s*#\s*interpreter: (?P<n>\d+) passed, 0 skipped",
        "CONTRIBUTING.md": r"\((\d+) tests as of (20\d\d-\d\d-\d\d)\)",
    }
    dates = set()
    for name, pattern in claims.items():
        m = re.search(pattern, (ROOT / name).read_text(encoding="utf-8"))
        assert m, name
        groups = m.groupdict() or dict(zip(("n", "date"), m.groups()))
        assert int(groups["n"]) == collected, (name, groups["n"], collected)
        dates.add(groups["date"])
    assert len(dates) == 1, dates  # one number, one date, in every file
