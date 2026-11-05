"""
Fairness audit for credit decisions.

Definitions follow Hardt, Price & Srebro (2016) and Barocas, Hardt & Narayanan,
*Fairness and Machine Learning*, in the LENDING framing: the favourable
outcome is the decision the API returns as "Approved" (``creditlens.decision``:
score >= 600), and an applicant is "qualified" if they repaid.

Gated criteria — each is reported as a worst-pair point estimate, a confidence
interval, the per-group sample sizes behind it, and a three-state verdict:

==================  =================================================================  =========
criterion           worst group pair (i, j) of                                          passes if
==================  =================================================================  =========
demographic parity  |P(approved | i) - P(approved | j)|                                  < 0.10
equal opportunity   |P(approved | repaid, i) - P(approved | repaid, j)|                  < 0.05
equalized odds      |Δ P(approved | repaid)| + |Δ P(approved | defaulted)|               < 0.10
disparate impact    min(P(approved | i), P(approved | j)) / max(...)                     >= 0.80
==================  =================================================================  =========

Reported, not gated: predictive parity (gap in P(default | not approved)) and
calibration within groups (observed default rate minus mean predicted
P(default), per group, with a normal-approximation interval that is
Bonferroni-adjusted over the groups, so the per-group intervals hold
simultaneously at 95%). Calibration is deliberately not a gate:
when base rates differ across groups, calibration within groups and error-rate
parity cannot all hold at once outside degenerate cases (Kleinberg,
Mullainathan & Raghavan 2016; Chouldechova 2017), so gating on both would
make every audit fail by construction. The report shows the trade-off instead.

Uncertainty and verdicts
------------------------
Every criterion depends only on each group's 2x2 table of (approved, not
approved) x (repaid, defaulted). Intervals come from a stratified bootstrap of
the audit set: each group's table is redrawn ``Multinomial(n_g, observed cell
shares)`` — exactly a with-replacement resample of that group's rows — 20,000
times (seeded, so a report is deterministic). Every interval is built from the
SIGNED differences behind a criterion, whose percentile intervals are
Bonferroni-adjusted so the criterion's intervals hold *simultaneously* at 95%:

* Demographic parity and equal opportunity resample one signed difference per
  pair (98.33% each for three groups) and map its interval to the absolute gap.
* Disparate impact does the same with the signed log approval ratio.
* Equalized odds is a sum of two absolute gaps, so it takes two signed
  differences per pair (99.17% each for three groups: Bonferroni over 3 pairs x
  2 gaps). Each interval is mapped to its absolute gap and the bounds are added,
  so the lower bound reaches 0 whenever both signed intervals span 0. This is
  conservative: the two gaps rarely sit at their extremes together, so coverage
  runs above 95%, and a verdict on it is decided later than a tighter interval
  would decide it.

The equalized-odds interval does not bootstrap the sum of absolute gaps itself: noise
only ever raises that sum, so on a fair model such an interval would sit wholly above
the truth (README "Fairness Metrics Reference" has the coverage measured). 20,000
replicates keep the Monte-Carlo error of a bound near 0.001, so a verdict near its
gate does not flip with the bootstrap seed alone.

* PASS — every pair's interval lies entirely on the passing side.
* FAIL — at least one pair's interval lies entirely on the failing side.
* INCONCLUSIVE — otherwise: the interval straddles the threshold. A small group
  is one reason, and a true gap close to the threshold is another, which takes a
  very large sample to decide. The demo's equalized odds shows both, separately:
  at 2,000 audited the verdict is mostly sample size (a model with no bias reads
  it too), and the injected group-C bias sits near the gate on a large population
  (README "Fairness Metrics Reference"). A pair is also
  INCONCLUSIVE, whatever its interval says, when a group has fewer than
  MIN_GROUP_N (30) applicants or fewer than MIN_OUTCOME_N (10) of an outcome
  the criterion conditions on (the bootstrap can degenerate there), and every
  criterion is INCONCLUSIVE when there are fewer than two groups — each with
  the reason in ``FairnessCheck.reason``.

The overall verdict is FAIL if any criterion fails, PASS if all four pass, and
INCONCLUSIVE otherwise. Each criterion is a 95% statement; the combination of
four is not itself one: nothing adjusts across criteria, so when all four
truly pass the overall verdict can read FAIL by chance up to 20% of the time
(Bonferroni, 4 x 5%; about 10% when each interval's tails are symmetric). The
equalized-odds POINT estimate keeps the upward bias of a sum of absolute gaps:
at small sizes it reads high on a fair model, so read the interval, as the verdict
does. The intervals cover sampling variability of the audit
set only — not training variability: retraining on another seed also moves the
point estimates.

Scope: these are screening diagnostics. ECOA / Regulation B and the Fair
Housing Act prohibit discrimination in credit, but neither prescribes these
metrics or these thresholds, and passing them is not evidence of legal
compliance. The 0.80 disparate-impact screen is borrowed from the EEOC's
Uniform Guidelines on Employee Selection Procedures (29 C.F.R. § 1607.4(D)),
which govern employment selection, not lending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

from creditlens.decision import (
    APPROVE_MIN_SCORE, REVIEW_MIN_SCORE, is_approved, probability_to_score,
)

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
VERDICTS = (PASS, FAIL, INCONCLUSIVE)

# Below these sizes a group pair cannot be judged (its verdict is INCONCLUSIVE,
# with the reason): the stratified bootstrap can degenerate — if every member
# of a tiny class shares one decision, every resample does too, and a
# zero-width interval would PASS or FAIL on nothing. A group needs MIN_GROUP_N
# applicants, and MIN_OUTCOME_N of each outcome a criterion conditions on
# (repaid for equal opportunity; repaid and defaulted for equalized odds).
MIN_GROUP_N = 30
MIN_OUTCOME_N = 10
# Which outcome counts each criterion conditions on.
_CONDITIONS = {
    "demographic_parity": (), "disparate_impact": (),
    "equal_opportunity": ("repaid",), "equalized_odds": ("repaid", "defaulted"),
}

CHECK_DEFINITIONS = {
    "demographic_parity": "max over group pairs of |P(approved | i) - P(approved | j)|",
    "equal_opportunity": (
        "max over group pairs of |P(approved | repaid, i) - P(approved | repaid, j)| "
        "(Hardt et al. 2016, lending framing)"
    ),
    "equalized_odds": (
        "max over group pairs of |Δ P(approved | repaid)| + |Δ P(approved | defaulted)|"
    ),
    "disparate_impact": (
        "min over group pairs of min(P(approved | i), P(approved | j)) / max(...) "
        "(0.80 screen borrowed from employment's 4/5ths guideline)"
    ),
}


@dataclass
class GroupMetrics:
    """Outcomes for a single demographic group.

    Rates are ``nan`` when undefined (e.g. no defaulters in the group): a
    silent 0 would feed a fake gap into the audit.
    """

    group: str
    n_samples: int
    n_positive: int                # defaulters (label 1)
    approval_rate: float           # P(decision == "Approved")
    true_positive_rate: float      # P(not approved | defaulted)
    false_positive_rate: float     # P(not approved | repaid)
    false_negative_rate: float     # P(approved | defaulted)
    precision: float               # P(defaulted | not approved)
    auc_roc: float
    avg_score: float
    default_rate: float            # actual default rate in group
    n_approved: int = 0
    approval_rate_repaid: float = float("nan")     # P(approved | repaid) — equal opportunity
    approval_rate_defaulted: float = float("nan")  # P(approved | defaulted)
    mean_predicted_default: float = float("nan")
    calibration_in_the_large: float = float("nan")  # default_rate - mean_predicted_default
    calibration_ci: tuple[float, float] = (float("nan"), float("nan"))

    @property
    def n_repaid(self) -> int:
        return self.n_samples - self.n_positive


@dataclass
class PairGap:
    """One group pair's estimate for one criterion."""

    groups: tuple[str, str]
    estimate: float                # gap (or ratio, for disparate impact)
    ci_low: float
    ci_high: float
    verdict: str
    reason: str = ""               # why it cannot be judged, if it cannot


@dataclass
class FairnessCheck:
    """One gated criterion: worst-pair estimate, its interval, and the verdict."""

    name: str
    definition: str
    threshold: float
    passes_when: str               # "value < threshold" | "value >= threshold"
    value: float                   # worst pair's point estimate
    worst_pair: tuple[str, str] | None
    ci_low: float                  # worst pair's simultaneous interval
    ci_high: float
    verdict: str
    pairs: list[PairGap] = field(default_factory=list)
    reason: str = ""               # why the criterion is not decided, if data is too thin


def equalized_odds_gap(tprs: list[float], fprs: list[float]) -> float:
    """
    Standard equalized-odds gap: the maximum over all group *pairs* of the
    combined true-/false-positive-rate discrepancy for that pair,

        max_{(i, j)}  ( |TPR_i - TPR_j| + |FPR_i - FPR_j| ).

    Equalized odds (Hardt et al. 2016) requires TPR *and* FPR to match across
    groups, so the gap is measured per pair and the worst pair reported. This
    differs from ``(max TPR - min TPR) + (max FPR - min FPR)``: that sum can add
    a TPR spread from one pair to an FPR spread from a *different* pair, only
    ever over-stating the true worst-pair discrepancy. ``tprs`` and ``fprs`` are
    aligned per group (same order). The gap is the same in the default-flagging
    and the approval framing (each rate is one minus its counterpart). With
    fewer than two groups the gap is 0; pairs with an undefined rate are
    skipped, and if no pair is defined the gap is ``nan``.
    """
    if len(tprs) < 2:
        return 0.0
    gaps = [
        abs(tprs[i] - tprs[j]) + abs(fprs[i] - fprs[j])
        for i, j in combinations(range(len(tprs)), 2)
    ]
    gaps = [g for g in gaps if not np.isnan(g)]
    return max(gaps) if gaps else float("nan")


@dataclass
class FairnessReport:
    """Complete fairness audit report."""

    group_metrics: dict[str, GroupMetrics]
    demographic_parity_diff: float         # worst pair |Δ P(approved)|
    equalized_odds_diff: float             # worst pair |Δ P(app|repaid)| + |Δ P(app|defaulted)|
    equal_opportunity_diff: float          # worst pair |Δ P(approved | repaid)|
    predictive_parity_diff: float          # worst pair |Δ P(defaulted | not approved)| (not gated)
    disparate_impact_ratio: float          # worst pair min/max approval rate

    # True only when the criterion's verdict is PASS (an INCONCLUSIVE
    # criterion does not pass). See ``checks`` for the three-state verdicts.
    passes_demographic_parity: bool
    passes_equal_opportunity: bool
    passes_disparate_impact: bool
    passes_equalized_odds: bool

    recommendations: list[str] = field(default_factory=list)
    checks: dict[str, FairnessCheck] = field(default_factory=dict)
    confidence: float = 0.95
    n_bootstrap: int = 0
    decision_rule: str = ""

    @property
    def overall_verdict(self) -> str:
        verdicts = [c.verdict for c in self.checks.values()]
        if not verdicts:  # a report built without intervals: point flags only
            verdicts = [PASS if ok else FAIL for ok in (
                self.passes_demographic_parity, self.passes_equal_opportunity,
                self.passes_disparate_impact, self.passes_equalized_odds,
            )]
        if FAIL in verdicts:
            return FAIL
        return PASS if all(v == PASS for v in verdicts) else INCONCLUSIVE

    @property
    def overall_pass(self) -> bool:
        return self.overall_verdict == PASS

    def __repr__(self) -> str:
        return (
            f"FairnessReport({self.overall_verdict} | "
            f"dem_parity_diff={self.demographic_parity_diff:.3f} | "
            f"disparate_impact={self.disparate_impact_ratio:.3f})"
        )


def _safe_div(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _cell_rates(cells: np.ndarray) -> dict[str, np.ndarray]:
    """Rates from (..., 4) cell counts ordered (approved&repaid,
    approved&defaulted, denied&repaid, denied&defaulted)."""
    ar, ad, dr, dd = (cells[..., k].astype(float) for k in range(4))
    return {
        "approval": _safe_div(ar + ad, ar + ad + dr + dd),
        "approved_repaid": _safe_div(ar, ar + dr),
        "approved_defaulted": _safe_div(ad, ad + dd),
        "defaulted_denied": _safe_div(dd, dr + dd),
    }


def _abs_interval(lo: float, hi: float) -> tuple[float, float]:
    """Interval for |Δ| implied by an interval for a signed Δ."""
    if np.isnan(lo) or np.isnan(hi):
        return float("nan"), float("nan")
    if lo >= 0:
        return lo, hi
    if hi <= 0:
        return -hi, -lo
    return 0.0, max(-lo, hi)


def _verdict(lo: float, hi: float, threshold: float, higher_is_better: bool) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return INCONCLUSIVE
    if higher_is_better:  # passes when value >= threshold
        return PASS if lo >= threshold else FAIL if hi < threshold else INCONCLUSIVE
    return PASS if hi < threshold else FAIL if lo >= threshold else INCONCLUSIVE


class FairnessAuditor:
    """
    Audits credit decisions for demographic disparities.

    Parameters
    ----------
    threshold : float | None
        ``None`` (default): audit the policy the API actually applies — an
        applicant is approved iff ``probability_to_score(p) >= APPROVE_MIN_SCORE``
        (``creditlens.decision``). Pass a probability only to audit a
        *hypothetical* cutoff: approved iff ``P(default) < threshold``.
    protected_attribute : str
        Column name for demographic group (default 'demographic_group').
    confidence : float
        Simultaneous coverage of each criterion's pairwise intervals.
    n_bootstrap, bootstrap_seed : int
        Bootstrap replicates and the seed that makes a report deterministic.

    Examples
    --------
    >>> auditor = FairnessAuditor()            # audits the served policy
    >>> report = auditor.audit(y_true, y_pred_proba, groups)
    >>> print(report)
    """

    def __init__(
        self,
        threshold: float | None = None,
        protected_attribute: str = "demographic_group",
        demographic_parity_threshold: float = 0.10,
        equal_opportunity_threshold: float = 0.05,
        disparate_impact_threshold: float = 0.80,
        equalized_odds_threshold: float = 0.10,
        confidence: float = 0.95,
        n_bootstrap: int = 20000,
        bootstrap_seed: int = 0,
    ) -> None:
        self.threshold = threshold
        self.protected_attribute = protected_attribute
        self.dem_parity_threshold = demographic_parity_threshold
        self.equal_opp_threshold = equal_opportunity_threshold
        self.di_threshold = disparate_impact_threshold
        self.eq_odds_threshold = equalized_odds_threshold
        self.confidence = confidence
        self.n_bootstrap = n_bootstrap
        self.bootstrap_seed = bootstrap_seed

    def _decision_rule(self) -> str:
        if self.threshold is None:
            return f"approved = score >= {APPROVE_MIN_SCORE} (the /score policy)"
        return f"approved = P(default) < {self.threshold} (hypothetical cutoff)"

    def audit(
        self,
        y_true: np.ndarray | pd.Series,
        y_proba: np.ndarray,
        groups: np.ndarray | pd.Series,
    ) -> FairnessReport:
        """
        Run the complete fairness audit on the FULL audit set (every group, no
        slicing — subsetting rows changes what the criteria mean).

        Parameters
        ----------
        y_true : array-like
            True default labels (0/1).
        y_proba : array-like
            Predicted default probabilities.
        groups : array-like
            Demographic group labels for each sample.

        Returns
        -------
        FairnessReport
        """
        from sklearn.metrics import roc_auc_score

        y_true = np.asarray(y_true).astype(int)
        y_proba = np.asarray(y_proba, dtype=float)
        groups = np.asarray(groups).astype(str)
        if self.threshold is None:
            approved = is_approved(probability_to_score(y_proba))
        else:
            approved = y_proba < self.threshold

        names = [str(g) for g in np.unique(groups)]
        cells: dict[str, np.ndarray] = {}
        group_metrics: dict[str, GroupMetrics] = {}
        # Calibration intervals hold simultaneously over the groups (Bonferroni),
        # as the gated criteria's do over group pairs: three groups each flagged
        # at 95% would raise a false alarm about 14% of the time.
        z_cal = NormalDist().inv_cdf(1 - (1 - self.confidence) / (2 * max(len(names), 1)))

        for g in names:
            mask = groups == g
            yt, ap, yp = y_true[mask], approved[mask], y_proba[mask]
            c = np.array([
                np.sum(ap & (yt == 0)), np.sum(ap & (yt == 1)),
                np.sum(~ap & (yt == 0)), np.sum(~ap & (yt == 1)),
            ])
            cells[g] = c
            r = {k: float(v) for k, v in _cell_rates(c).items()}
            n = int(mask.sum())
            resid = yt - yp
            cal = float(resid.mean())
            half = z_cal * float(resid.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
            try:
                auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else float("nan")
            except ValueError:
                auc = float("nan")
            group_metrics[g] = GroupMetrics(
                group=g,
                n_samples=n,
                n_positive=int(yt.sum()),
                approval_rate=r["approval"],
                true_positive_rate=1.0 - r["approved_defaulted"],
                false_positive_rate=1.0 - r["approved_repaid"],
                false_negative_rate=r["approved_defaulted"],
                precision=r["defaulted_denied"],
                auc_roc=float(auc),
                avg_score=float((1 - yp).mean() * 1000),
                default_rate=float(yt.mean()),
                n_approved=int(ap.sum()),
                approval_rate_repaid=r["approved_repaid"],
                approval_rate_defaulted=r["approved_defaulted"],
                mean_predicted_default=float(yp.mean()),
                calibration_in_the_large=cal,
                calibration_ci=(float(cal - half), float(cal + half)),
            )

        checks = self._checks(names, cells)
        pred_parity = [
            abs(group_metrics[i].precision - group_metrics[j].precision)
            for i, j in combinations(names, 2)
        ]
        pred_parity = [d for d in pred_parity if not np.isnan(d)]

        report = FairnessReport(
            group_metrics=group_metrics,
            demographic_parity_diff=checks["demographic_parity"].value,
            equalized_odds_diff=checks["equalized_odds"].value,
            equal_opportunity_diff=checks["equal_opportunity"].value,
            predictive_parity_diff=max(pred_parity) if pred_parity else float("nan"),
            disparate_impact_ratio=checks["disparate_impact"].value,
            passes_demographic_parity=checks["demographic_parity"].verdict == PASS,
            passes_equal_opportunity=checks["equal_opportunity"].verdict == PASS,
            passes_disparate_impact=checks["disparate_impact"].verdict == PASS,
            passes_equalized_odds=checks["equalized_odds"].verdict == PASS,
            checks=checks,
            confidence=self.confidence,
            n_bootstrap=self.n_bootstrap,
            decision_rule=self._decision_rule(),
        )
        report.recommendations = self._generate_recommendations(report)
        return report

    def _checks(self, names: list[str], cells: dict[str, np.ndarray]) -> dict[str, FairnessCheck]:
        rng = np.random.default_rng(self.bootstrap_seed)
        observed = {g: _cell_rates(cells[g]) for g in names}
        boot = {}
        for g in names:
            n = int(cells[g].sum())
            draws = rng.multinomial(n, cells[g] / n, size=self.n_bootstrap) if n else np.zeros((self.n_bootstrap, 4))
            boot[g] = _cell_rates(draws)
        pairs = list(combinations(names, 2))
        alpha = (1.0 - self.confidence) / max(len(pairs), 1)

        def interval(stat: np.ndarray, level_alpha: float = alpha) -> tuple[float, float]:
            stat = stat[~np.isnan(stat)]
            if len(stat) < 0.5 * self.n_bootstrap:
                return float("nan"), float("nan")
            lo, hi = np.quantile(stat, [level_alpha / 2, 1 - level_alpha / 2])
            return float(lo), float(hi)

        def odds_interval(a, b) -> tuple[float, float]:
            # Both signed gaps' intervals hold together (Bonferroni over the two),
            # so the sum of absolute gaps lies between the sums of their bounds.
            repaid = _abs_interval(*interval(a["approved_repaid"] - b["approved_repaid"], alpha / 2))
            defaulted = _abs_interval(*interval(a["approved_defaulted"] - b["approved_defaulted"], alpha / 2))
            return repaid[0] + defaulted[0], repaid[1] + defaulted[1]

        def log_rate(rate):
            return np.log(np.maximum(rate, 1e-12))

        specs = {
            # name: (threshold, higher_is_better, point(i,j), interval(i,j))
            "demographic_parity": (self.dem_parity_threshold, False,
                lambda a, b: abs(a["approval"] - b["approval"]),
                lambda a, b: _abs_interval(*interval(a["approval"] - b["approval"]))),
            "equal_opportunity": (self.equal_opp_threshold, False,
                lambda a, b: abs(a["approved_repaid"] - b["approved_repaid"]),
                lambda a, b: _abs_interval(*interval(a["approved_repaid"] - b["approved_repaid"]))),
            "equalized_odds": (self.eq_odds_threshold, False,
                lambda a, b: abs(a["approved_repaid"] - b["approved_repaid"])
                + abs(a["approved_defaulted"] - b["approved_defaulted"]),
                odds_interval),
            "disparate_impact": (self.di_threshold, True,
                lambda a, b: float(np.exp(-abs(log_rate(a["approval"]) - log_rate(b["approval"])))),
                lambda a, b: tuple(float(np.exp(-x)) for x in reversed(
                    _abs_interval(*interval(log_rate(a["approval"]) - log_rate(b["approval"])))))),
        }

        def thin(g: str, name: str) -> str:
            ar, ad, dr, dd = (int(x) for x in cells[g])
            n, repaid, defaulted = ar + ad + dr + dd, ar + dr, ad + dd
            if n < MIN_GROUP_N:
                return f"group {g} has {n} applicants (< {MIN_GROUP_N})"
            counts = {"repaid": repaid, "defaulted": defaulted}
            for outcome in _CONDITIONS[name]:
                if counts[outcome] < MIN_OUTCOME_N:
                    return f"group {g} has {counts[outcome]} {outcome} (< {MIN_OUTCOME_N})"
            return ""

        checks: dict[str, FairnessCheck] = {}
        for name, (threshold, higher_better, point, ci) in specs.items():
            results = []
            for i, j in pairs:
                est = float(point(observed[i], observed[j]))
                lo, hi = ci(boot[i], boot[j]) if not np.isnan(est) else (float("nan"), float("nan"))
                why = "; ".join(r for r in (thin(i, name), thin(j, name)) if r)
                verdict = INCONCLUSIVE if why else _verdict(lo, hi, threshold, higher_better)
                results.append(PairGap((i, j), est, lo, hi, verdict, why))
            defined = [p for p in results if not np.isnan(p.estimate)]
            if defined:
                worst = (min if higher_better else max)(defined, key=lambda p: p.estimate)
                value, worst_pair, ci_low, ci_high = worst.estimate, worst.groups, worst.ci_low, worst.ci_high
            else:
                value, worst_pair, ci_low, ci_high = float("nan"), None, float("nan"), float("nan")
            verdicts = [p.verdict for p in results]
            reason = "; ".join(dict.fromkeys(p.reason for p in results if p.reason))
            if not results:
                # One group (or none): there is nothing to compare, and a wrong or
                # constant group column must not read as a fair model.
                verdict = INCONCLUSIVE
                reason = f"the audit needs at least two groups; found {len(names)}"
            elif FAIL in verdicts:
                verdict = FAIL
            elif all(v == PASS for v in verdicts):
                verdict = PASS
            else:
                verdict = INCONCLUSIVE
            checks[name] = FairnessCheck(
                name=name,
                definition=CHECK_DEFINITIONS[name],
                threshold=threshold,
                passes_when="value >= threshold" if higher_better else "value < threshold",
                value=float(value),
                worst_pair=worst_pair,
                ci_low=ci_low,
                ci_high=ci_high,
                verdict=verdict,
                pairs=results,
                reason=reason,
            )
        return checks

    def compute_score_distribution(
        self,
        scores: np.ndarray,
        groups: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compare score distributions per group.

        Returns
        -------
        pd.DataFrame with percentile breakdown per group.
        """
        rows = []
        for g in np.unique(groups):
            mask = groups == g
            group_scores = scores[mask]
            rows.append({
                "group": str(g),
                "n": int(mask.sum()),
                "mean_score": float(np.mean(group_scores)),
                "median_score": float(np.median(group_scores)),
                "p10": float(np.percentile(group_scores, 10)),
                "p25": float(np.percentile(group_scores, 25)),
                "p75": float(np.percentile(group_scores, 75)),
                "p90": float(np.percentile(group_scores, 90)),
                "pct_approved": float((group_scores >= APPROVE_MIN_SCORE).mean()),
                "pct_denied": float((group_scores < REVIEW_MIN_SCORE).mean()),
            })
        return pd.DataFrame(rows).set_index("group")

    def mitigation_reweighting(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
    ) -> np.ndarray:
        """
        Compute sample weights for fairness-aware retraining.

        Implements Kamiran & Calders (2012) *reweighing*: each (group, label)
        cell is weighted by ``P(group) * P(label) / P(group, label)`` — the
        ratio of the frequency the cell would have if group and label were
        independent to its observed frequency. Over-represented cells (e.g. a
        group disproportionately labelled as defaulting) are down-weighted and
        under-represented cells up-weighted, so the reweighted data carries no
        statistical association between group membership and the label.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Labels (1 = default).
        groups : pd.Series
            Group labels.

        Returns
        -------
        np.ndarray
            Sample weights array (same length as X), to pass as
            ``LogisticScorer.fit(X, y, sample_weight=weights)``.

        Scope: reweighing removes a statistical association between group and
        LABEL in the training data. It does not correct an input that is
        measured differently for one group (e.g. the opt-in
        ``group_c_fico_shift``, an understated bureau score): that bias lives in
        the features, and the reweighted model still sees it.
        """
        weights = np.ones(len(X))
        y_arr = np.asarray(y)
        g_arr = np.asarray(groups)
        n = len(y_arr)

        for g in np.unique(g_arr):
            g_mask = g_arr == g
            p_group = g_mask.mean()
            for label in (0, 1):
                cell = g_mask & (y_arr == label)
                observed = cell.mean()
                if observed == 0:
                    continue
                p_label = float((y_arr == label).mean())
                # Expected frequency under independence / observed frequency.
                weights[cell] = (p_group * p_label) / observed

        # Normalize so the total weight equals the sample count.
        weights *= n / weights.sum()
        return weights

    # ------------------------------------------------------------------
    # private
    # ------------------------------------------------------------------

    def _generate_recommendations(self, report: FairnessReport) -> list[str]:
        level = f"{report.confidence:.0%}"
        labels = {
            "demographic_parity": "Demographic parity",
            "equal_opportunity": "Equal opportunity",
            "equalized_odds": "Equalized odds",
            "disparate_impact": "Disparate impact",
        }
        recs: list[str] = []
        if len(report.group_metrics) < 2:
            found = ", ".join(f"'{g}'" for g in report.group_metrics) or "none"
            return [f"No fairness conclusion is possible: the audit needs at least two groups and "
                    f"found {len(report.group_metrics)} ({found}). Check the group column: a wrong "
                    "or constant column cannot show a disparity."]
        for name, check in report.checks.items():
            if check.verdict == PASS:
                continue
            if check.worst_pair is None:
                recs.append(f"{labels[name]}: not estimable — a group lacks the outcomes "
                            "this criterion conditions on.")
                continue
            if check.verdict == INCONCLUSIVE and check.reason:
                thin = sum(1 for p in check.pairs if p.reason)
                recs.append(f"{labels[name]} is INCONCLUSIVE: too little data to judge {thin} of "
                            f"{len(check.pairs)} group pair(s) ({check.reason}).")
                continue
            i, j = check.worst_pair
            sizes = ", ".join(
                f"{g}: n={report.group_metrics[g].n_samples}, "
                f"{report.group_metrics[g].n_positive} defaulted"
                for g in (i, j)
            )
            measured = (f"{check.value:.3f} for {i} vs {j} "
                        f"({level} CI {check.ci_low:.3f}–{check.ci_high:.3f}; {sizes})")
            if check.verdict == FAIL:
                failing = ", ".join(
                    f"{a} vs {b} {p.estimate:.3f} (CI {p.ci_low:.3f}–{p.ci_high:.3f})"
                    for p in check.pairs if p.verdict == FAIL for a, b in [p.groups]
                )
                recs.append(
                    f"{labels[name]} FAILS: {measured}; threshold {check.threshold:g}. "
                    f"Beyond it with {level} confidence: {failing}. "
                    + ("Find which inputs drive the gap (the per-applicant attributions "
                       "of /explain) and whether any is measured differently for one group "
                       "— a biased input, such as the opt-in demo's understated group-C "
                       "FICO, has to be corrected in the data. Reweighing "
                       "(mitigation_reweighting, passed as LogisticScorer.fit(sample_weight=)) "
                       "removes only a group-label association in the training data and "
                       "cannot repair a biased input."
                       if name in ("demographic_parity", "disparate_impact")
                       else "Error rates differ by group: find which inputs drive the gap "
                            "(the attributions of /explain) and whether any is measured "
                            "differently for one group. This library implements no "
                            "post-processing (group-specific thresholds).")
                )
            else:
                recs.append(
                    f"{labels[name]} is INCONCLUSIVE: {measured}; the interval straddles the "
                    f"threshold {check.threshold:g}. Too little data may explain that, and so may "
                    "a true gap close to the threshold, which no sample size makes clear-cut: "
                    "compare a larger held-out draw and other training seeds before drawing a "
                    "conclusion."
                )
        flagged = False
        for g, m in report.group_metrics.items():
            lo, hi = m.calibration_ci
            if not np.isnan(lo) and (lo > 0 or hi < 0):
                flagged = True
                direction = "under-predicts" if lo > 0 else "over-predicts"
                recs.append(
                    f"Calibration (reported, not gated): the model {direction} group {g}'s "
                    f"risk — observed default rate {m.default_rate:.3f} vs mean predicted "
                    f"{m.mean_predicted_default:.3f} (gap {m.calibration_in_the_large:+.3f}; "
                    f"simultaneous {level} CI {lo:+.4f} to {hi:+.4f}; n={m.n_samples})."
                )
        gaps = {g: m for g, m in report.group_metrics.items() if not np.isnan(m.calibration_ci[0])}
        if not flagged and gaps:
            g, m = max(gaps.items(), key=lambda kv: abs(kv[1].calibration_in_the_large))
            recs.append(
                f"Calibration (reported, not gated): the largest calibration gap is group {g}'s — "
                f"observed default rate {m.default_rate:.3f} vs mean predicted "
                f"{m.mean_predicted_default:.3f} (gap {m.calibration_in_the_large:+.3f}; simultaneous "
                f"{level} CI {m.calibration_ci[0]:+.4f} to {m.calibration_ci[1]:+.4f}; n={m.n_samples}). "
                "This audit set cannot separate it from zero."
            )
        if not recs:
            recs.append("No criterion violated: every group pair's interval lies inside its threshold.")
        return recs
