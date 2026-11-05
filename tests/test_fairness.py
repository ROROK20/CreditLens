"""Tests for fairness metrics and bias detection."""

import numpy as np
import pandas as pd
import pytest

from creditlens.fairness.metrics import (
    FairnessAuditor,
    FairnessReport,
    GroupMetrics,
    equalized_odds_gap,
)


def make_balanced_groups(n: int = 600, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate y_true, y_proba, groups with known fair properties."""
    rng = np.random.default_rng(seed)
    groups = np.array(["A"] * (n // 3) + ["B"] * (n // 3) + ["C"] * (n // 3))
    y_true = rng.binomial(1, 0.15, n)
    y_proba = np.clip(y_true * 0.6 + rng.normal(0.15, 0.1, n), 0, 1)
    return y_true, y_proba, groups


def make_biased_groups(n: int = 600, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data with strong group C bias."""
    rng = np.random.default_rng(seed)
    groups = np.array(["A"] * (n // 3) + ["B"] * (n // 3) + ["C"] * (n // 3))
    y_true = np.zeros(n, dtype=int)
    y_proba = np.full(n, 0.10)

    # Group C gets much higher predicted probabilities (biased model)
    c_mask = groups == "C"
    y_proba[c_mask] = 0.60

    y_true = (y_proba + rng.normal(0, 0.05, n) > 0.3).astype(int)
    return y_true, y_proba, groups


def _from_tables(tables: dict[str, tuple[int, int, int, int]]):
    """Build (y_true, y_proba, groups) realising exact per-group 2x2 tables of
    (approved&repaid, approved&defaulted, denied&repaid, denied&defaulted).
    P(default) 0.10 -> score 900 (approved under the served policy AND under a
    0.20 cutoff); 0.90 -> score 100 (denied under both)."""
    y, p, g = [], [], []
    for name, (ar, ad, dr, dd) in tables.items():
        for count, label, proba in ((ar, 0, 0.1), (ad, 1, 0.1), (dr, 0, 0.9), (dd, 1, 0.9)):
            y += [label] * count
            p += [proba] * count
            g += [name] * count
    return np.array(y), np.array(p), np.array(g)


@pytest.fixture
def auditor():
    return FairnessAuditor(threshold=0.20)


class TestFairnessAuditor:
    def test_audit_returns_report(self, auditor):
        y_true, y_proba, groups = make_balanced_groups()
        report = auditor.audit(y_true, y_proba, groups)
        assert isinstance(report, FairnessReport)

    def test_group_metrics_present(self, auditor):
        y_true, y_proba, groups = make_balanced_groups()
        report = auditor.audit(y_true, y_proba, groups)
        assert "A" in report.group_metrics
        assert "B" in report.group_metrics
        assert "C" in report.group_metrics

    def test_dem_parity_diff_nonnegative(self, auditor):
        y_true, y_proba, groups = make_balanced_groups()
        report = auditor.audit(y_true, y_proba, groups)
        assert report.demographic_parity_diff >= 0

    def test_disparate_impact_between_zero_one(self, auditor):
        y_true, y_proba, groups = make_balanced_groups()
        report = auditor.audit(y_true, y_proba, groups)
        assert 0 <= report.disparate_impact_ratio <= 1

    def test_biased_model_fails_checks(self, auditor):
        y_true, y_proba, groups = make_biased_groups()
        report = auditor.audit(y_true, y_proba, groups)
        # Biased model should fail at least one fairness check
        assert not report.overall_pass or report.disparate_impact_ratio < 0.9

    def test_balanced_model_passes(self, auditor):
        """A perfectly fair model should pass all checks."""
        n = 300
        y_true = np.array([0] * 255 + [1] * 45)
        y_proba = np.full(n, 0.15)  # same prediction for all
        groups = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
        report = auditor.audit(y_true, y_proba, groups)
        # Demographic parity should be 0 (all groups get same score)
        assert report.demographic_parity_diff < 1e-6

    def test_recommendations_non_empty(self, auditor):
        y_true, y_proba, groups = make_biased_groups()
        report = auditor.audit(y_true, y_proba, groups)
        assert len(report.recommendations) > 0

    def test_score_distribution_returns_dataframe(self, auditor):
        y_true, y_proba, groups = make_balanced_groups()
        scores = np.round((1 - y_proba) * 1000).astype(int)
        dist = auditor.compute_score_distribution(scores, groups)
        assert isinstance(dist, pd.DataFrame)
        assert "mean_score" in dist.columns
        assert "pct_approved" in dist.columns

    def test_mitigation_reweighting_shape(self, auditor):
        from creditlens.data.generator import generate_credit_dataset, DatasetConfig
        df = generate_credit_dataset(DatasetConfig(n_samples=500))
        from creditlens.features.engineering import FeatureEngineer
        engineer = FeatureEngineer()
        X, y = engineer.fit_transform(df)
        groups = df["demographic_group"]
        weights = auditor.mitigation_reweighting(X, y, groups)
        assert len(weights) == len(X)
        assert (weights > 0).all()
        # Normalized: total weight equals the sample count.
        assert weights.sum() == pytest.approx(len(X))

    def test_mitigation_reweighting_removes_group_label_association(self, auditor):
        # Kamiran & Calders reweighing: after weighting, P_w(y=1 | group) must be
        # the same for every group (group ⟂ label under the weighted measure).
        rng = np.random.default_rng(7)
        n = 900
        groups = pd.Series(["A"] * 300 + ["B"] * 300 + ["C"] * 300)
        # Group C defaults 3x as often — a strong group/label association.
        base = np.array([0.08] * 300 + [0.10] * 300 + [0.30] * 300)
        y = pd.Series(rng.binomial(1, base))
        X = pd.DataFrame({"f": rng.normal(size=n)})

        w = auditor.mitigation_reweighting(X, y, groups)

        weighted_rates = []
        for g in ["A", "B", "C"]:
            mask = (groups == g).values
            rate = np.sum(w[mask] * y.values[mask]) / np.sum(w[mask])
            weighted_rates.append(rate)
        # All weighted default rates equal (up to float error).
        assert max(weighted_rates) - min(weighted_rates) < 1e-9

    def test_equalized_odds_gates_overall_pass(self, auditor):
        # Every group here has both outcomes, so the gap is real and estimable (the
        # degenerate case is pinned separately below as "not estimable").
        y_true, y_proba, groups = _from_tables({
            # (approved&repaid, approved&defaulted, denied&repaid, denied&defaulted)
            "A": (1500, 60, 100, 340),
            "B": (1500, 60, 100, 340),
            "C": (1350, 240, 250, 160),
        })
        report = auditor.audit(y_true, y_proba, groups)
        assert isinstance(report.passes_equalized_odds, bool)
        # overall_pass must reflect ALL four flags, including equalized odds.
        assert report.overall_pass == all([
            report.passes_demographic_parity,
            report.passes_equal_opportunity,
            report.passes_disparate_impact,
            report.passes_equalized_odds,
        ])
        # A vs C: |0.9375 - 0.84375| + |0.15 - 0.60| = 0.09375 + 0.45.
        assert report.equalized_odds_diff == pytest.approx(0.09375 + 0.45)
        assert report.checks["equalized_odds"].verdict == "FAIL"
        assert report.checks["equalized_odds"].ci_low >= 0.10
        assert report.passes_equalized_odds is False
        assert report.overall_pass is False

    def test_degenerate_groups_make_equalized_odds_not_estimable(self, auditor):
        """No defaulters in A/B and no repayers in C: P(approved | defaulted) is
        undefined for A and B, P(approved | repaid) for C. The gap must be
        reported as not estimable, never as a fake gap from 0/0 scored as 0."""
        y_true, y_proba, groups = make_biased_groups()
        report = auditor.audit(y_true, y_proba, groups)
        assert np.isnan(report.group_metrics["A"].approval_rate_defaulted)
        assert np.isnan(report.group_metrics["C"].approval_rate_repaid)
        assert np.isnan(report.equalized_odds_diff)
        assert report.checks["equalized_odds"].verdict == "INCONCLUSIVE"
        assert any(r.startswith("Equalized odds: not estimable") for r in report.recommendations)
        # Demographic parity is still measurable here, and fails decisively.
        assert report.checks["demographic_parity"].verdict == "FAIL"
        assert report.overall_verdict == "FAIL"

    def test_equalized_odds_is_per_pair_max_not_sum_of_ranges(self, auditor):
        """Pin the STANDARD equalized-odds definition and distinguish it from the
        old ``(max TPR − min TPR) + (max FPR − min FPR)`` sum-of-ranges bug.

        Three groups whose TPR range and FPR range are achieved by *different*
        pairs, so the two formulas give different numbers:

            group   TPR    FPR
              A     0.90   0.10
              B     0.50   0.30
              C     0.70   0.60

        TPR range = 0.90 − 0.50 = 0.40  (pair A,B)
        FPR range = 0.60 − 0.10 = 0.50  (pair A,C)   ← different pair
        sum of ranges                  = 0.40 + 0.50 = 0.90
        per-pair |ΔTPR| + |ΔFPR|: (A,B)=0.60  (A,C)=0.70  (B,C)=0.50
        standard gap = max over pairs = 0.70  (pair A,C)

        The reported gap must be 0.70, not the sum of ranges' 0.90.
        """
        # Realise the exact TPR/FPR above. 100 defaulters + 100 non-defaulters
        # per group; feed y_proba of 0.9 for a predicted-default and 0.0
        # otherwise so the auditor's `y_proba >= threshold(0.20)` reproduces the
        # intended predictions exactly (1 = predicted default).
        specs = {"A": (0.90, 0.10), "B": (0.50, 0.30), "C": (0.70, 0.60)}
        y_true_parts, proba_parts, group_parts = [], [], []
        for g, (tpr, fpr) in specs.items():
            n_def = n_nondef = 100
            tp = round(tpr * n_def)          # defaulters predicted-default
            fp = round(fpr * n_nondef)       # non-defaulters predicted-default
            # defaulters block: tp predicted-1 then (n_def-tp) predicted-0
            y_true_parts += [1] * n_def + [0] * n_nondef
            proba_parts += [0.9] * tp + [0.0] * (n_def - tp)
            proba_parts += [0.9] * fp + [0.0] * (n_nondef - fp)
            group_parts += [g] * (n_def + n_nondef)

        y_true = np.array(y_true_parts)
        y_proba = np.array(proba_parts)
        groups = np.array(group_parts)

        report = auditor.audit(y_true, y_proba, groups)

        # Sanity: the fixture realised the intended per-group rates.
        assert report.group_metrics["A"].true_positive_rate == pytest.approx(0.90)
        assert report.group_metrics["C"].false_positive_rate == pytest.approx(0.60)

        # The STANDARD per-pair-max gap.
        assert report.equalized_odds_diff == pytest.approx(0.70)
        # The sum of ranges would report 0.90, so this must be strictly less.
        sum_of_ranges = (0.90 - 0.50) + (0.60 - 0.10)
        assert sum_of_ranges == pytest.approx(0.90)
        assert report.equalized_odds_diff < sum_of_ranges - 1e-9

    def test_equalized_odds_gap_helper_matches_hand_computation(self):
        """The pure helper is directly hand-checkable and pair-order invariant."""
        tprs = [0.90, 0.50, 0.70]
        fprs = [0.10, 0.30, 0.60]
        assert equalized_odds_gap(tprs, fprs) == pytest.approx(0.70)
        # Single group (or none) → no disparity possible.
        assert equalized_odds_gap([0.8], [0.2]) == 0.0
        assert equalized_odds_gap([], []) == 0.0
        # Identical groups → zero gap.
        assert equalized_odds_gap([0.5, 0.5], [0.3, 0.3]) == pytest.approx(0.0)

    def test_equalized_odds_gap_never_exceeds_sum_of_ranges(self):
        """Mathematical invariant: the per-pair-max standard gap is always ≤ the
        sum of ranges, which can only over-state it."""
        rng = np.random.default_rng(123)
        for _ in range(200):
            k = int(rng.integers(2, 6))
            tprs = rng.random(k).tolist()
            fprs = rng.random(k).tolist()
            gap = equalized_odds_gap(tprs, fprs)
            sum_of_ranges = (max(tprs) - min(tprs)) + (max(fprs) - min(fprs))
            assert gap <= sum_of_ranges + 1e-12

    def test_equal_opportunity_diff_nonnegative(self, auditor):
        y_true, y_proba, groups = make_balanced_groups()
        report = auditor.audit(y_true, y_proba, groups)
        assert report.equal_opportunity_diff >= 0

    def test_group_approval_rates_between_zero_one(self, auditor):
        y_true, y_proba, groups = make_balanced_groups()
        report = auditor.audit(y_true, y_proba, groups)
        for g, m in report.group_metrics.items():
            assert 0 <= m.approval_rate <= 1, f"Group {g}: approval_rate={m.approval_rate}"


class TestGroupMetrics:
    def test_repr(self):
        gm = GroupMetrics(
            group="A", n_samples=100, n_positive=15,
            approval_rate=0.85, true_positive_rate=0.70,
            false_positive_rate=0.10, false_negative_rate=0.30,
            precision=0.65, auc_roc=0.82, avg_score=720.0, default_rate=0.15
        )
        # Dataclass auto-repr should include the class name and the group
        r = repr(gm)
        assert "GroupMetrics" in r
        assert "A" in r

    def test_fairness_report_repr(self, auditor):
        # The repr must state the report's ACTUAL verdict and numbers.
        y_true, y_proba, groups = make_biased_groups()
        report = auditor.audit(y_true, y_proba, groups)
        assert repr(report) == "FairnessReport(FAIL | dem_parity_diff=1.000 | disparate_impact=0.000)"

        # ...including the third state, which a PASS/FAIL repr would misstate.
        y_true, y_proba, groups = _from_tables({"A": (1615, 150, 85, 150), "C": (30, 3, 4, 3)})
        report = FairnessAuditor().audit(y_true, y_proba, groups)
        assert report.overall_verdict == "INCONCLUSIVE"
        assert repr(report) == (
            f"FairnessReport(INCONCLUSIVE | dem_parity_diff={report.demographic_parity_diff:.3f} | "
            f"disparate_impact={report.disparate_impact_ratio:.3f})"
        )


class TestStandardDefinitionsAndUncertainty:
    """Equal opportunity in its standard lending form, per-group sample sizes,
    bootstrap intervals, and a verdict that only says FAIL (or PASS) when the
    interval clears the threshold."""

    def test_equal_opportunity_is_approval_rate_among_repaid(self):
        # Same P(approved | repaid) = 0.95 in both groups; very different rates
        # of catching DEFAULTERS (A denies 50%, B 90%).
        y, p, g = _from_tables({"A": (760, 100, 40, 100), "B": (760, 20, 40, 180)})
        report = FairnessAuditor().audit(y, p, g)
        assert report.group_metrics["A"].approval_rate_repaid == pytest.approx(0.95)
        assert report.group_metrics["B"].approval_rate_repaid == pytest.approx(0.95)
        # Standard (Hardt et al.) gap: 0. The gap in defaulters caught, 0.40, is not this.
        assert report.equal_opportunity_diff == pytest.approx(0.0)
        assert "repaid" in report.checks["equal_opportunity"].definition

    def test_equal_opportunity_gap_among_repaid_is_measured(self):
        # Defaulters treated identically; repayers approved 0.95 vs 0.80.
        y, p, g = _from_tables({"A": (1520, 100, 80, 300), "B": (1280, 100, 320, 300)})
        report = FairnessAuditor().audit(y, p, g)
        check = report.checks["equal_opportunity"]
        assert check.value == pytest.approx(0.15)
        assert check.worst_pair == ("A", "B")
        assert check.ci_low <= 0.15 <= check.ci_high
        assert check.ci_low >= 0.05 and check.verdict == "FAIL"

    def test_reports_per_group_sample_sizes(self):
        y, p, g = _from_tables({"A": (700, 30, 50, 120), "C": (60, 4, 9, 7)})
        m = FairnessAuditor().audit(y, p, g).group_metrics
        assert (m["A"].n_samples, m["A"].n_repaid, m["A"].n_positive, m["A"].n_approved) == (900, 750, 150, 730)
        assert (m["C"].n_samples, m["C"].n_repaid, m["C"].n_positive, m["C"].n_approved) == (80, 69, 11, 64)

    def test_small_group_gap_above_threshold_is_inconclusive_not_fail(self):
        """C has 34 repayers. Its point gap (0.068) exceeds the 0.05 threshold, and
        the interval straddles it, so it is not a FAIL."""
        y, p, g = _from_tables({"A": (1615, 150, 85, 150), "C": (30, 3, 4, 3)})
        report = FairnessAuditor().audit(y, p, g)
        check = report.checks["equal_opportunity"]
        assert check.value > 0.05
        assert check.ci_low < 0.05 < check.ci_high
        assert check.verdict == "INCONCLUSIVE"
        assert report.passes_equal_opportunity is False  # INCONCLUSIVE does not pass
        assert report.overall_verdict == "INCONCLUSIVE"
        assert any("INCONCLUSIVE" in r and "n=40" in r for r in report.recommendations)

    def test_identical_large_groups_pass_every_criterion(self):
        table = (4275, 75, 225, 675)
        y, p, g = _from_tables({"A": table, "B": table, "C": table})
        report = FairnessAuditor().audit(y, p, g)
        for name, check in report.checks.items():
            assert check.verdict == "PASS", (name, check.ci_low, check.ci_high)
        assert report.checks["disparate_impact"].ci_low >= 0.80
        assert report.overall_verdict == "PASS" and report.overall_pass is True

    def test_intervals_cover_the_true_gap(self):
        """Frequentist check of the intervals themselves: draw 300 audit sets
        from a known population (approval 0.90 vs 0.85, a true parity gap of
        0.05) and count how often the reported 95% interval contains 0.05."""
        rng = np.random.default_rng(2026)
        auditor = FairnessAuditor(n_bootstrap=2000)
        groups = np.array(["A"] * 600 + ["B"] * 200)
        rate = np.where(groups == "A", 0.90, 0.85)
        hits = 0
        for _ in range(300):
            approved = rng.random(800) < rate
            y = rng.binomial(1, 0.15, 800)
            check = auditor.audit(y, np.where(approved, 0.1, 0.9), groups).checks["demographic_parity"]
            hits += check.ci_low <= 0.05 <= check.ci_high
        assert hits / 300 >= 0.93

    def test_undefined_rate_is_nan_not_zero(self):
        # C has no defaulters: P(approved | defaulted) is undefined for C.
        y, p, g = _from_tables({"A": (900, 20, 30, 50), "B": (900, 20, 30, 50), "C": (950, 0, 50, 0)})
        report = FairnessAuditor().audit(y, p, g)
        assert np.isnan(report.group_metrics["C"].approval_rate_defaulted)
        assert np.isnan(report.group_metrics["C"].true_positive_rate)
        pairs = {pg.groups: pg for pg in report.checks["equalized_odds"].pairs}
        assert np.isnan(pairs[("A", "C")].estimate) and pairs[("A", "C")].verdict == "INCONCLUSIVE"
        # A and B are identical, so the only DEFINED pair has gap 0 — still the
        # criterion cannot PASS while two of its pairs are unmeasurable.
        assert report.equalized_odds_diff == pytest.approx(0.0)
        assert report.checks["equalized_odds"].verdict == "INCONCLUSIVE"

    def test_calibration_within_groups_is_reported(self):
        """Observed default rate minus mean predicted P(default), per group."""
        rng = np.random.default_rng(5)
        groups = np.array(["A"] * 2000 + ["C"] * 1000)
        y = np.concatenate([rng.binomial(1, 0.20, 2000), rng.binomial(1, 0.40, 1000)])
        report = FairnessAuditor().audit(y, np.full(3000, 0.20), groups)
        a, c = report.group_metrics["A"], report.group_metrics["C"]
        assert a.mean_predicted_default == pytest.approx(0.20)
        assert a.calibration_ci[0] < 0 < a.calibration_ci[1]
        assert c.calibration_in_the_large == pytest.approx(c.default_rate - 0.20)
        assert c.calibration_ci[0] > 0.15
        assert any("under-predicts group C" in r for r in report.recommendations)

    def test_report_is_deterministic(self):
        y, p, g = make_balanced_groups()
        one, two = FairnessAuditor().audit(y, p, g), FairnessAuditor().audit(y, p, g)
        for name in one.checks:
            assert (one.checks[name].ci_low, one.checks[name].ci_high) == (
                two.checks[name].ci_low, two.checks[name].ci_high)


class TestTooLittleDataToJudge:
    """A one-group audit has no pairs to compare, so a wrong or constant group column
    must not read as fair, and a tiny group must not PASS or FAIL on a degenerate
    bootstrap."""

    def test_one_group_is_never_a_pass(self):
        rng = np.random.default_rng(0)
        report = FairnessAuditor(n_bootstrap=2000).audit(
            rng.integers(0, 2, 500), rng.uniform(0, 1, 500), ["A"] * 500)
        assert report.overall_verdict == "INCONCLUSIVE" and report.overall_pass is False
        for name, check in report.checks.items():
            assert check.verdict == "INCONCLUSIVE", name
            assert np.isnan(check.value), name
            assert "at least two groups" in check.reason, name
        assert len(report.recommendations) == 1
        assert report.recommendations[0].startswith("No fairness conclusion is possible")

    def test_a_tiny_group_cannot_pass_or_fail(self):
        # Two group-C applicants, both approved: every bootstrap resample of C
        # is "100% approved", so its interval has zero width, which must not make
        # the 0.5 gap a FAIL.
        y, p, g = _from_tables({"A": (225, 25, 225, 25), "C": (2, 0, 0, 0)})
        report = FairnessAuditor(n_bootstrap=2000).audit(y, p, g)
        for name, check in report.checks.items():
            assert check.verdict == "INCONCLUSIVE", name
            assert "group C has 2 applicants (< 30)" in check.reason, name
        assert any("too little data" in r for r in report.recommendations)

    def test_a_thin_outcome_class_blocks_only_the_criteria_that_condition_on_it(self):
        # Group C: 100 applicants but only 3 defaulters.
        y, p, g = _from_tables({"A": (380, 40, 20, 60), "C": (90, 1, 7, 2)})
        report = FairnessAuditor(n_bootstrap=2000).audit(y, p, g)
        assert report.checks["equalized_odds"].verdict == "INCONCLUSIVE"
        assert "group C has 3 defaulted (< 10)" in report.checks["equalized_odds"].reason
        # Approval rates (84% vs 91%) are judged normally: n is large enough.
        assert report.checks["demographic_parity"].reason == ""
        assert report.checks["equal_opportunity"].reason == ""


def test_fail_advice_offers_only_what_the_code_can_do():
    """A demographic-parity FAIL names reweighing as fit(sample_weight=) makes it possible,
    says it cannot repair a biased input such as the opt-in understated group-C FICO, and
    offers no post-processing, which does not exist."""
    y, p, g = _from_tables({"A": (1520, 100, 80, 300), "B": (1000, 100, 600, 300)})
    report = FairnessAuditor(n_bootstrap=2000).audit(y, p, g)
    dp = next(r for r in report.recommendations if r.startswith("Demographic parity FAILS"))
    assert "LogisticScorer.fit(sample_weight=)" in dp
    assert "cannot repair a biased input" in dp
    eo = next(r for r in report.recommendations if r.startswith("Equal opportunity FAILS"))
    assert "implements no post-processing" in eo
    assert not any("consider group-aware post-processing" in r for r in report.recommendations)
