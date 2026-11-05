"""What the fairness headline may say.

At the audit's 2,000 applicants equalized odds' INCONCLUSIVE is mostly sample size
(tests/test_equalized_odds_interval.py). Separately, the generator's `group_c_bias`
makes group C default more often at the same features, the model cannot see it, and
on a large fresh draw equalized odds sits within about 0.01 of its 0.10 gate. These
tests re-derive the population values, how the verdict moves over seeds, and what the
report says about group C.
"""

from __future__ import annotations

import os
import re
import threading
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from creditlens.api import app as api
from creditlens.data.generator import DatasetConfig, generate_credit_dataset, train_test_split
from creditlens.fairness.metrics import FairnessAuditor
from creditlens.features.engineering import FeatureEngineer

ROOT = Path(__file__).resolve().parents[1]
DRAWS = range(2026, 2036)  # ten fresh draws, the first of them what the API audits


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        api._startup_train()
        yield c


@pytest.fixture(scope="module")
def report(client):
    return client.get("/fairness/report").json()


def _readme() -> str:
    return re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))


def _fit(n: int, seed: int):
    from creditlens.models.scorer import LogisticScorer

    df = generate_credit_dataset(DatasetConfig(n_samples=n, seed=seed))
    train, test = train_test_split(df, test_size=0.20, seed=seed)
    engineer = FeatureEngineer(scale=True)
    X, y = engineer.fit_transform(train)
    return LogisticScorer().fit(X, y), engineer, test


def _audit(model, engineer, frame):
    X, y = engineer.transform(frame)
    return FairnessAuditor().audit(y.values, model.predict_proba(X)[:, 1], frame["demographic_group"].values)


def ten_draws() -> dict:
    model, engineer, _ = _fit(api.DEFAULT_TRAIN_SAMPLES, api.DEFAULT_SEED)
    reports = [_audit(model, engineer, generate_credit_dataset(DatasetConfig(n_samples=400_000, seed=s)))
               for s in DRAWS]
    odds = [r.checks["equalized_odds"] for r in reports]
    return {
        "mean": float(np.mean([c.value for c in odds])),
        "low": min(c.value for c in odds),
        "high": max(c.value for c in odds),
        "verdicts": [c.verdict for c in odds],
        "gap": float(np.mean([r.group_metrics["C"].default_rate - r.group_metrics["A"].default_rate for r in reports])),
        "calibration": float(np.mean([r.group_metrics["C"].calibration_in_the_large for r in reports])),
    }


def seed_sweep() -> dict:
    rows = [_audit(m, e, t) for m, e, t in (_fit(api.DEFAULT_TRAIN_SAMPLES, s) for s in range(1, 20))]
    verdict = lambda name: [r.checks[name].verdict for r in rows]  # noqa: E731
    odds = [r.checks["equalized_odds"].value for r in rows]
    return {
        "overall": [r.overall_verdict for r in rows],
        "dp": verdict("demographic_parity"), "eo": verdict("equal_opportunity"),
        "odds": verdict("equalized_odds"), "di": verdict("disparate_impact"),
        "odds_low": min(odds), "odds_high": max(odds),
        "calibration_flagged": sum(r.group_metrics["C"].calibration_ci[0] > 0 for r in rows),
    }


def test_a_straddling_interval_is_not_blamed_on_sample_size(report):
    """The straddling message does not send the reader to audit more applicants, on a
    gate the population itself sits at."""
    assert not any("Audit more held-out applicants" in r for r in report["recommendations"])
    pick = next(r for r in report["recommendations"] if "straddles the threshold" in r)
    assert "a true gap close to the threshold" in pick


def test_the_report_says_group_c_is_under_predicted(report):
    """The generator's comment says the audit surfaces the bias, so the recommendations
    name it."""
    assert any("under-predicts group C's risk" in r for r in report["recommendations"])


def test_the_report_carries_the_large_draw(report):
    population = report["population"]
    assert population["n_samples"] == 400_000 and population["seed"] == DRAWS[0]
    groups = population["group_metrics"]
    assert sum(g["n_samples"] for g in groups.values()) == 400_000
    low, high = groups["C"]["calibration_ci"]
    assert 0.05 < low < groups["C"]["calibration_in_the_large"] < high
    for name in ("A", "B"):
        assert abs(groups[name]["calibration_in_the_large"]) < 0.02
    # The audit set's own calibration stays in the report, with its interval.
    for g in report["group_metrics"].values():
        assert g["calibration_ci"][0] <= g["calibration_in_the_large"] <= g["calibration_ci"][1]


def test_the_readme_states_the_population_values(report):
    text = _readme()
    p = report["population"]
    odds = p["checks"]["equalized_odds"]
    c, a = p["group_metrics"]["C"], p["group_metrics"]["A"]
    for claim in (
        f"On 400,000 fresh applicants (seed {DRAWS[0]}) equalized odds reads {odds['value']:.3f} "
        f"(CI {odds['ci'][0]:.3f}–{odds['ci'][1]:.3f})",
        f"group C defaults at {c['default_rate']:.1%} against {c['mean_predicted_default']:.1%} predicted",
        f"(gap {c['calibration_in_the_large'] * 100:+.1f} points, CI "
        f"{c['calibration_ci'][0] * 100:+.1f} to {c['calibration_ci'][1] * 100:+.1f})",
        f"approves {c['approval_rate_defaulted']:.1%} of group C's defaulters against "
        f"{a['approval_rate_defaulted']:.1%} of group A's",
    ):
        assert claim in text, claim
    for name, label in (("demographic_parity", "demographic parity"), ("equal_opportunity", "equal opportunity"),
                        ("disparate_impact", "disparate impact")):
        check = p["checks"][name]
        assert check["verdict"] == "PASS", name
        assert f"{label} {check['value']:.3f}" in text, name


def test_the_readme_states_the_ten_draws():
    d = ten_draws()
    text = _readme()
    inconclusive, failed = d["verdicts"].count("INCONCLUSIVE"), d["verdicts"].count("FAIL")
    assert inconclusive + failed == 10
    assert 0.09 < d["mean"] < 0.13  # at the gate, not far from it either way
    for claim in (
        f"Over ten such draws (seeds {DRAWS[0]}–{DRAWS[-1]}) it averages {d['mean']:.3f} "
        f"(range {d['low']:.3f}–{d['high']:.3f}; {inconclusive} INCONCLUSIVE, {failed} FAIL)",
        f"group C's default rate exceeds group A's by {d['gap'] * 100:.1f} points",
    ):
        assert claim in text, claim


def test_the_readme_states_how_the_verdict_moves_over_seeds():
    s = seed_sweep()
    text = _readme()
    assert s["overall"].count("INCONCLUSIVE") == 19
    assert (f"Over training seeds 1–19 at the API's size the overall verdict reads INCONCLUSIVE "
            f"{s['overall'].count('INCONCLUSIVE')} times") in text
    assert (f"equalized odds ranges {s['odds_low']:.3f}–{s['odds_high']:.3f}, never decided") in text
    assert set(s["odds"]) == {"INCONCLUSIVE"}
    assert f"equal opportunity PASSes on {s['eo'].count('PASS')} of the 19" in text
    assert f"demographic parity on {s['dp'].count('PASS')}" in text
    assert f"group C's calibration interval excludes zero on {s['calibration_flagged']} of the 19" in text
    for seed in (42, 7):
        model, engineer, test = _fit(50_000, seed)
        r = _audit(model, engineer, test)
        c = r.checks["equalized_odds"]
        assert f"seed {seed} reads {r.overall_verdict} ({c.value:.3f})" in text, seed


def test_the_readme_states_the_verdict_over_a_wider_window_of_seeds():
    """Seed 88 reads FAIL; seeds 33 and 73 read INCONCLUSIVE. The full window takes
    minutes: CREDITLENS_FULL_SEED_SWEEP=1 re-runs all of it."""
    text = _readme()
    assert ("over seeds 1–100 it reads INCONCLUSIVE 99 times and FAIL once (seed 88, on equalized odds: "
            "0.325, CI 0.111–0.527)") in text
    assert "equalized odds ranges 0.035–0.325 over that window" in text
    window = range(1, 101) if os.environ.get("CREDITLENS_FULL_SEED_SWEEP") else (33, 73, 88)
    rows = {s: _audit(*_fit(api.DEFAULT_TRAIN_SAMPLES, s)) for s in window}
    assert [s for s, r in rows.items() if r.overall_verdict == "FAIL"] == [88]
    assert all(rows[s].checks["equalized_odds"].verdict == "INCONCLUSIVE" for s in (33, 73))
    assert rows[88].checks["equalized_odds"].verdict == "FAIL"
    odds = rows[88].checks["equalized_odds"]
    assert (round(odds.value, 3), round(odds.ci_low, 3), round(odds.ci_high, 3)) == (0.325, 0.111, 0.527)
    odds = rows[33].checks["equalized_odds"]
    assert f"Seed 33 reads {odds.value:.3f}, CI {odds.ci_low:.3f}–{odds.ci_high:.3f}" in text
    if len(rows) == 100:
        assert sum(r.overall_verdict == "INCONCLUSIVE" for r in rows.values()) == 99
        values = [r.checks["equalized_odds"].value for r in rows.values()]
        assert (round(min(values), 3), round(max(values), 3)) == (0.035, 0.325)


def test_the_report_carries_predictive_parity(report):
    """The README lists it as reported, not gated, and /fairness/report carries it."""
    model, engineer, test = _fit(api.DEFAULT_TRAIN_SAMPLES, api.DEFAULT_SEED)
    audit = _audit(model, engineer, test)
    assert report["predictive_parity_diff"] == round(audit.predictive_parity_diff, 4)
    assert report["population"]["predictive_parity_diff"] is not None


def test_the_large_draw_audit_holds_little_memory(client, monkeypatch):
    """Traced allocations, so the host's load cannot move this. README "Deploy" has what
    a first page load takes in the API's container."""
    monkeypatch.setitem(api._model_state, "fairness_population", None)
    tracemalloc.start()
    try:
        api._population_report()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 250 * 1024 * 1024, f"{peak / 1048576:.0f} MB"


def test_callers_during_the_first_audit_share_one_computation(client, monkeypatch):
    """The page asks for the report on load, so callers during the first computation
    share it. The second caller waits on the lock however long it takes to get there."""
    from creditlens.data import generator

    real = generator.generate_credit_dataset
    calls, entered, release = [], threading.Event(), threading.Event()

    def held(config):
        calls.append(config.seed)
        entered.set()
        assert release.wait(120)
        return real(config)

    monkeypatch.setattr(generator, "generate_credit_dataset", held)
    monkeypatch.setattr(api, "POPULATION_N", 2000)
    monkeypatch.setitem(api._model_state, "fairness_population", None)
    results = []
    threads = [threading.Thread(target=lambda: results.append(api._population_report())) for _ in range(2)]
    threads[0].start()
    assert entered.wait(120)
    threads[1].start()
    threads[1].join(2)
    release.set()
    for thread in threads:
        thread.join(300)
    assert len(results) == 2 and results[0] is results[1]
    assert len(calls) == 1


def test_the_deploy_notes_state_the_measured_memory():
    readme, blueprint = _readme(), (ROOT / "render.yaml").read_text(encoding="utf-8")
    measured = re.search(r"peaks at about (\d+) MB in that image", readme)
    assert measured and int(measured.group(1)) < 500
    assert f"{measured.group(1)} MB" in blueprint
    assert "check your plan's memory" in readme and "check your plan's memory" in blueprint


def test_an_unflagged_calibration_gap_is_still_named():
    """On the API's own audit set group C's interval can straddle zero, as it does
    on seed 42. The report states the gap rather than saying nothing."""
    model, engineer, test = _fit(api.DEFAULT_TRAIN_SAMPLES, api.DEFAULT_SEED)
    recs = _audit(model, engineer, test).recommendations
    assert any("largest calibration gap is group C's" in r and "cannot separate it from zero" in r for r in recs)
