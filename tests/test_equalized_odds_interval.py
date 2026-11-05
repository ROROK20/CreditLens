"""Equalized odds' interval covers the truth, at the audit's own size.

Noise only ever raises a sum of absolute gaps, so a percentile bootstrap of that sum sits
above the truth on a fair model. The interval comes from the two signed gaps' simultaneous
intervals, with the bounds added.

CREDITLENS_FULL_SEED_SWEEP=1 re-runs the README's whole coverage table, separation figures
and seed comparison (minutes). They are deterministic, and fail here if one stops
reproducing.
"""

from __future__ import annotations

import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from creditlens.data.generator import DatasetConfig, generate_credit_dataset, train_test_split
from creditlens.decision import is_approved, probability_to_score
from creditlens.fairness.metrics import FairnessAuditor
from creditlens.features.engineering import FeatureEngineer
from creditlens.models.scorer import LogisticScorer

ROOT = Path(__file__).resolve().parents[1]
BIASES = {"no bias": 0.0, "default bias": 0.08}
FULL = bool(os.environ.get("CREDITLENS_FULL_SEED_SWEEP"))


def _tables(tables: dict[str, tuple[int, int, int, int]]):
    """(approved&repaid, approved&defaulted, denied&repaid, denied&defaulted) per group."""
    y, p, g = [], [], []
    for name, (ar, ad, dr, dd) in tables.items():
        for count, label, proba in ((ar, 0, 0.1), (ad, 1, 0.1), (dr, 0, 0.9), (dd, 1, 0.9)):
            y += [label] * count
            p += [proba] * count
            g += [name] * count
    return np.array(y), np.array(p), np.array(g)


def _odds(tables, **kwargs):
    return FairnessAuditor(**kwargs).audit(*_tables(tables)).checks["equalized_odds"]


def _fit(seed: int, bias: float, n: int = 10_000):
    df = generate_credit_dataset(DatasetConfig(n_samples=n, seed=seed, group_c_bias=bias))
    train, test = train_test_split(df, seed=seed)
    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(train)
    return LogisticScorer().fit(X, y), engineer, test


def _proba(model, engineer, frame) -> np.ndarray:
    return np.concatenate([model.predict_proba(engineer.transform(frame.iloc[i:i + 20000])[0])[:, 1]
                           for i in range(0, len(frame), 20000)])


def _pool(bias: float, seeds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model, engineer, _ = _fit(42, bias)
    frames = []
    for seed in seeds:
        q = generate_credit_dataset(DatasetConfig(n_samples=400_000, seed=seed, group_c_bias=bias))
        frames.append(pd.DataFrame({"p": _proba(model, engineer, q), "y": q["default"].to_numpy(),
                                    "g": q["demographic_group"].to_numpy()}))
    pool = pd.concat(frames, ignore_index=True)
    return pool["y"].to_numpy(), pool["p"].to_numpy(), pool["g"].to_numpy()


def _truth(y, p, g) -> dict:
    audit = FairnessAuditor(n_bootstrap=200).audit(y, p, g)
    return {pair.groups: pair.estimate for pair in audit.checks["equalized_odds"].pairs}


class TestTheBounds:
    def test_the_lower_bound_reaches_zero_when_both_signed_intervals_span_zero(self):
        """Two groups with identical tables: the interval reaches 0, which no bootstrap
        of the sum of absolute gaps can."""
        check = _odds({"A": (700, 60, 40, 60), "B": (700, 60, 40, 60)})
        assert check.value == 0.0
        assert check.ci_low == 0.0
        assert check.ci_high > 0.0

    def test_the_interval_is_the_signed_gaps_intervals_with_their_bounds_added(self):
        """Rebuilt from a separate bootstrap: one pair, so each gap's interval is at 97.5%."""
        tables = {"A": (700, 80, 40, 60), "B": (300, 20, 40, 40)}
        check = _odds(tables, n_bootstrap=40_000)
        rng = np.random.default_rng(99)
        draws = {g: rng.multinomial(sum(t), np.array(t) / sum(t), size=40_000) for g, t in tables.items()}
        gaps = []
        for approved, other in ((0, 2), (1, 3)):
            a = draws["A"][:, approved] / (draws["A"][:, approved] + draws["A"][:, other])
            b = draws["B"][:, approved] / (draws["B"][:, approved] + draws["B"][:, other])
            lo, hi = np.quantile(a - b, [0.0125, 0.9875])
            gaps.append((0.0 if lo < 0 < hi else min(abs(lo), abs(hi)), max(abs(lo), abs(hi))))
        assert check.ci_low == pytest.approx(gaps[0][0] + gaps[1][0], abs=0.01)
        assert check.ci_high == pytest.approx(gaps[0][1] + gaps[1][1], abs=0.01)
        assert check.ci_low <= check.value <= check.ci_high

    def test_a_clear_gap_on_one_rate_lifts_the_lower_bound_and_can_fail(self):
        check = _odds({"A": (900, 80, 50, 20), "B": (900, 20, 50, 80)})
        assert check.value == pytest.approx(0.60)
        assert check.ci_low >= 0.10 and check.verdict == "FAIL"

    def test_equal_rates_at_a_large_size_pass(self):
        check = _odds({"A": (50_000, 6_000, 3_000, 4_000), "B": (50_000, 6_000, 3_000, 4_000)}, n_bootstrap=4000)
        assert check.ci_high < 0.10 and check.verdict == "PASS"

    def test_a_thin_group_still_reads_inconclusive_with_a_nan_free_or_nan_interval(self):
        y, p, g = _tables({"A": (900, 20, 30, 50), "B": (900, 20, 30, 50), "C": (950, 0, 50, 0)})
        check = FairnessAuditor().audit(y, p, g).checks["equalized_odds"]
        pair = next(pg for pg in check.pairs if pg.groups == ("A", "C"))
        assert np.isnan(pair.ci_low) and np.isnan(pair.ci_high)
        assert pair.verdict == "INCONCLUSIVE"


@pytest.fixture(scope="module", params=list(BIASES))
def world(request):
    y, p, g = _pool(BIASES[request.param], [3000])
    return request.param, y, p, g, _truth(y, p, g)


def _audits(world, n: int, repeats: int, seed: int = 7):
    _, y, p, g, truth = world
    rng = np.random.default_rng(seed)
    covered = dict.fromkeys(truth, 0)
    verdicts = {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0}
    for r in range(repeats):
        ix = rng.integers(0, len(y), n)
        check = FairnessAuditor(n_bootstrap=2000, bootstrap_seed=r).audit(y[ix], p[ix], g[ix]).checks["equalized_odds"]
        verdicts[check.verdict] += 1
        for pair in check.pairs:
            covered[pair.groups] += pair.ci_low <= truth[pair.groups] <= pair.ci_high
    return {k: v / repeats for k, v in covered.items()}, verdicts


class TestCoverageAtTheAuditsSize:
    def test_every_pair_covers_its_true_gap(self, world):
        coverage, _ = _audits(world, 2000, 150)
        for pair, share in coverage.items():
            assert share >= 0.95, (world[0], pair, share)

    def test_a_small_audit_cannot_pass_either_model(self, world):
        _, verdicts = _audits(world, 2000, 60)
        assert verdicts["PASS"] == 0
        assert verdicts["INCONCLUSIVE"] >= 57, verdicts


def test_the_readme_and_the_page_say_what_the_verdict_is_about():
    readme = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    page = re.sub(r"\s+", " ", (ROOT / "frontend" / "app" / "page.tsx").read_text(encoding="utf-8"))
    assert "not only sample size" not in readme and "not only because" not in readme
    assert "not only a matter of sample size" not in page
    for text in (readme, page):
        assert "mostly sample size" in text and "with no injected bias" in text
    assert "At this size that verdict is mostly sample size" in readme
    assert "the injected bias sits near the gate on a large population" in readme


def test_a_model_with_no_bias_reads_the_same_verdict_at_this_size():
    """Seed 42 reads higher with no bias than with it, which is what no separation looks like."""
    readings = {}
    for label, bias in BIASES.items():
        model, engineer, test = _fit(42, bias)
        check = FairnessAuditor().audit(test["default"].to_numpy(), _proba(model, engineer, test),
                                        test["demographic_group"].to_numpy()).checks["equalized_odds"]
        assert check.verdict == "INCONCLUSIVE" and check.ci_low <= 0.01
        readings[label] = check.value
    readme = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    assert f"Seed 42 reads {readings['no bias']:.3f} with no bias and {readings['default bias']:.3f} with it" in readme


def test_the_readme_states_the_coverage_table():
    text = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    for label in BIASES:
        for n in ("2,000", "10,000", "50,000", "100,000"):
            assert f"| {label} | {n} | " in text
    if not FULL:
        return
    for label, bias in BIASES.items():
        y, p, g = _pool(bias, range(3000, 3004))
        truth = _truth(y, p, g)
        rng = np.random.default_rng(5)
        for n, repeats in ((2000, 300), (10_000, 200), (50_000, 100), (100_000, 50)):
            verdicts = {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0}
            covered = dict.fromkeys(truth, 0)
            readings = []
            for r in range(repeats):
                ix = rng.integers(0, len(y), n)
                check = FairnessAuditor(n_bootstrap=20_000, bootstrap_seed=r).audit(y[ix], p[ix], g[ix]).checks["equalized_odds"]
                verdicts[check.verdict] += 1
                readings.append(check.value)
                for pair in check.pairs:
                    covered[pair.groups] += pair.ci_low <= truth[pair.groups] <= pair.ci_high
            shares = " / ".join(f"{covered[pair] / repeats:.3f}" for pair in combinations("ABC", 2))
            row = (f"| {label} | {n:,} | {np.mean(readings):.3f} | {shares} | "
                   f"{verdicts['PASS']} / {verdicts['FAIL']} / {verdicts['INCONCLUSIVE']} |")
            assert row in text, row


def test_the_readme_states_the_audit_size_that_separates_them():
    text = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    for phrase in ("0 of 300 audits of 10,000", "14 of 300 at 20,000", "161 of 300 at 30,000", "287 of 300 at 50,000",
                   "200 of 200 at 100,000", "27 of 200 audits at 400,000", "151 of 200 at 1.6 million",
                   "199 of 200 at 3.2 million"):
        assert phrase in text, phrase
    if not FULL:
        return
    claims = {
        "no bias": {10_000: ("PASS", 0, 300), 20_000: ("PASS", 14, 300), 30_000: ("PASS", 161, 300),
                    50_000: ("PASS", 287, 300), 100_000: ("PASS", 200, 200)},
        "default bias": {400_000: ("FAIL", 27, 200), 1_600_000: ("FAIL", 151, 200), 3_200_000: ("FAIL", 199, 200)},
    }
    sizes = ((2000, 300), (5000, 300), (10_000, 300), (20_000, 300), (30_000, 300), (40_000, 300), (50_000, 300),
             (75_000, 200), (100_000, 200), (200_000, 200), (400_000, 200), (800_000, 200), (1_600_000, 200),
             (3_200_000, 200), (6_400_000, 100))
    for label, bias in BIASES.items():
        model, engineer, _ = _fit(42, bias)
        cells = {g: np.zeros(4) for g in "ABC"}
        for seed in range(3000, 3004):
            q = generate_credit_dataset(DatasetConfig(n_samples=400_000, seed=seed, group_c_bias=bias))
            approved = is_approved(probability_to_score(_proba(model, engineer, q)))
            y, g = q["default"].to_numpy(), q["demographic_group"].to_numpy()
            for name in "ABC":
                k = g == name
                cells[name] += [np.sum(approved[k] & (y[k] == 0)), np.sum(approved[k] & (y[k] == 1)),
                                np.sum(~approved[k] & (y[k] == 0)), np.sum(~approved[k] & (y[k] == 1))]
        names = list("ABC")
        shares = np.concatenate([cells[name] for name in names])
        shares /= shares.sum()
        rng = np.random.default_rng(11)
        for n, repeats in sizes:
            verdicts = {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0}
            for r in range(repeats):
                draw = rng.multinomial(n, shares)
                tables = {name: draw[4 * i:4 * i + 4] for i, name in enumerate(names)}
                verdicts[FairnessAuditor(bootstrap_seed=r)._checks(names, tables)["equalized_odds"].verdict] += 1
            if label == "default bias":
                assert verdicts["PASS"] == 0, n
            claim = claims[label].get(n)
            if claim:
                verdict, count, total = claim
                assert (verdicts[verdict], repeats) == (count, total), (label, n, verdicts)


def test_the_readme_states_the_no_bias_seed_comparison():
    text = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    assert "on all 100 of training seeds 1–100" in text
    if not FULL:
        return
    rows = {}
    for label, bias in BIASES.items():
        rows[label] = []
        for seed in range(1, 101):
            model, engineer, test = _fit(seed, bias)
            rows[label].append(FairnessAuditor().audit(test["default"].to_numpy(), _proba(model, engineer, test),
                                                       test["demographic_group"].to_numpy()).checks["equalized_odds"])
    assert all(c.verdict == "INCONCLUSIVE" for c in rows["no bias"])
    mean = lambda label, field: float(np.mean([getattr(c, field) for c in rows[label]]))  # noqa: E731
    assert (f"with a mean reading of {mean('no bias', 'value'):.3f} and a mean upper bound of "
            f"{mean('no bias', 'ci_high'):.3f}; with the bias the same seeds read "
            f"{mean('default bias', 'value'):.3f} and {mean('default bias', 'ci_high'):.3f}") in text
