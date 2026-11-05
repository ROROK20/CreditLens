"""Tests for the SHAP explainability layer.

Guards two regressions:
1. With the DEFAULT config (calibrate=True → CalibratedClassifierCV), the
   "fast TreeExplainer" path silently failed (the `.estimator` attribute is the
   UNFITTED template) and always fell back to the slow KernelExplainer.
2. TreeExplainer emits numpy.float32 SHAP values, which FastAPI/Pydantic cannot
   JSON-encode — explanations must carry native Python floats.
"""

from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")

from creditlens.data.generator import generate_credit_dataset, DatasetConfig, train_test_split
from creditlens.features.engineering import FeatureEngineer
from creditlens.models.scorer import XGBoostScorer
from creditlens.explainability.shap_explainer import SHAPExplainer


@contextmanager
def _seeded_global_rng(seed: int = 0):
    """shap's KernelExplainer samples coalitions from numpy's GLOBAL generator, so it is
    seeded here. Otherwise the reference these tests compare against differs on every run,
    and the mass check below can fail for reasons unrelated to the code."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _kernel_values(kernel, row) -> np.ndarray:
    """One row's KernelSHAP reference, the same whatever ran before it."""
    with _seeded_global_rng():
        return np.asarray(kernel.shap_values(row, nsamples=4096, l1_reg=False, silent=True))[0]


@pytest.fixture(scope="module")
def fitted_explainer():
    df = generate_credit_dataset(DatasetConfig(n_samples=1500, seed=7))
    train_df, test_df = train_test_split(df, test_size=0.25, seed=7)
    engineer = FeatureEngineer(scale=True)
    X_train, y_train = engineer.fit_transform(train_df)
    X_test, _ = engineer.transform(test_df)

    model = XGBoostScorer(n_estimators=40)  # default calibrate=True
    model.fit(X_train, y_train)

    explainer = SHAPExplainer(model, engineer.feature_names)
    explainer.fit(X_train.iloc[:50])  # small on purpose: KernelExplainer below costs per row
    return explainer, X_test


class TestSHAPExplainer:
    def test_default_config_explains_the_served_probability(self, fitted_explainer):
        # One path for every model: Shapley values of the served predict_proba against
        # a fixed background, with no silent fallback to guard against.
        from creditlens.explainability.shap_explainer import ServedShapleyExplainer

        explainer, _ = fitted_explainer
        assert isinstance(explainer._explainer, ServedShapleyExplainer)
        assert explainer.method == SHAPExplainer.METHOD
        assert explainer._explainer.background.shape == (50, len(explainer.feature_names))

    def test_individual_explanation_has_native_floats(self, fitted_explainer):
        explainer, X_test = fitted_explainer
        exp = explainer.explain_individual(X_test.iloc[[0]], applicant_id=1)
        assert isinstance(exp.base_value, float)
        assert not isinstance(exp.base_value, np.floating)
        for v in exp.shap_values.values():
            assert isinstance(v, float)
            assert not isinstance(v, np.floating)
        for factor in exp.top_positive_factors + exp.top_negative_factors:
            assert isinstance(factor["shap_value"], float)
            assert not isinstance(factor["shap_value"], np.floating)

    def test_explanation_is_directionally_complete(self, fitted_explainer):
        explainer, X_test = fitted_explainer
        exp = explainer.explain_individual(X_test.iloc[[0]], applicant_id=2)
        assert 0.0 <= exp.predicted_default_prob <= 1.0
        assert 0 <= exp.predicted_score <= 1000
        assert exp.decision in {"Approved", "Review", "Denied"}
        assert len(exp.shap_values) == len(explainer.feature_names)

    def test_successful_explanation_reports_ok_status(self, fitted_explainer):
        explainer, X_test = fitted_explainer
        exp = explainer.explain_individual(X_test.iloc[[0]], applicant_id=3)
        assert exp.status == "ok"
        assert exp.status_reason is None

    def test_shap_failure_yields_explicit_unavailable_not_silent_zeros(
        self, fitted_explainer, monkeypatch
    ):
        """A SHAP computation failure must surface an explicit degraded marker —
        NOT silently return zeroed attributions that read as 'no driving
        factors'. The score/decision (from the model, not SHAP) stay valid."""
        explainer, X_test = fitted_explainer

        def boom(*_a, **_k):
            raise RuntimeError("simulated SHAP kernel failure")

        # Force the underlying explainer's shap_values to fail.
        monkeypatch.setattr(explainer._explainer, "shap_values", boom)

        exp = explainer.explain_individual(X_test.iloc[[0]], applicant_id=99)

        # Explicit degraded status + reason, not a silent success.
        assert exp.status == "unavailable"
        assert exp.status_reason and "simulated SHAP kernel failure" in exp.status_reason
        # Factor lists are EMPTY (not zero-valued rows masquerading as factors).
        assert exp.top_positive_factors == []
        assert exp.top_negative_factors == []
        assert exp.shap_values == {}
        # Score/decision remain valid — they do not depend on SHAP.
        assert 0 <= exp.predicted_score <= 1000
        assert exp.decision in {"Approved", "Review", "Denied"}
        # The human-readable text names the degradation.
        assert "unavailable" in exp.explanation_text.lower()

    def test_global_shap_failure_raises_not_silent_zeros(
        self, fitted_explainer, monkeypatch
    ):
        """A global SHAP failure must raise, not silently return an all-zero
        matrix that would yield a meaningless feature-importance ranking."""
        explainer, X_test = fitted_explainer

        def boom(*_a, **_k):
            raise RuntimeError("simulated global SHAP failure")

        monkeypatch.setattr(explainer._explainer, "shap_values", boom)

        with pytest.raises(RuntimeError, match="Global SHAP computation failed"):
            explainer.explain_global(X_test.iloc[:20])


class TestExplainsTheServedScore:
    """The explanation must explain the probability the API serves, on every row, so it
    can never imply a different decision."""

    def test_base_plus_contributions_reconstruct_served_probability(self, fitted_explainer):
        explainer, X_test = fitted_explainer
        rows = X_test.iloc[:60]
        served = explainer.model.predict_proba(rows)[:, 1]
        for i in range(len(rows)):
            exp = explainer.explain_individual(rows.iloc[[i]], applicant_id=i)
            assert exp.status == "ok"
            rebuilt = exp.base_value + sum(exp.shap_values.values())
            assert rebuilt == pytest.approx(served[i], abs=1e-9), i

    def test_reconstruction_holds_for_the_uncalibrated_scorer(self):
        df = generate_credit_dataset(DatasetConfig(n_samples=800, seed=9))
        engineer = FeatureEngineer(scale=True)
        X, y = engineer.fit_transform(df)
        model = XGBoostScorer(n_estimators=30, calibrate=False).fit(X, y)
        explainer = SHAPExplainer(model, engineer.feature_names).fit(X.iloc[:100])
        assert explainer.method == SHAPExplainer.METHOD  # same estimator for every model
        served = model.predict_proba(X.iloc[:20])[:, 1]
        for i in range(20):
            exp = explainer.explain_individual(X.iloc[[i]])
            assert exp.base_value + sum(exp.shap_values.values()) == pytest.approx(served[i], abs=1e-9)

    def test_global_attributions_are_in_the_served_space(self, fitted_explainer):
        explainer, X_test = fitted_explainer
        glob = explainer.explain_global(X_test.iloc[:80])
        served = explainer.model.predict_proba(X_test.iloc[:80])[:, 1]
        rebuilt = explainer._explainer.expected_value + glob.shap_summary.sum(axis=1).to_numpy()
        np.testing.assert_allclose(rebuilt, served, atol=1e-9)


class TestExplanationText:
    """The text is built from THIS applicant's SHAP attribution: no generic tips, and no
    promise of a credit officer's call, since nothing in this project contacts anyone."""

    CANNED = ("credit officer", "business days", "emergency fund", "below 30%", "12+ months")

    def test_text_is_the_applicants_attribution_for_every_decision(self, fitted_explainer):
        from creditlens.decision import DECISION_REASONS, probability_to_score, score_to_decision

        explainer, X_test = fitted_explainer
        served = explainer.model.predict_proba(X_test)[:, 1]
        decisions = [score_to_decision(probability_to_score(p)) for p in served]
        for decision in ("Approved", "Review", "Denied"):
            assert decision in decisions, f"fixture has no {decision} applicant"
            i = decisions.index(decision)
            exp = explainer.explain_individual(X_test.iloc[[i]], applicant_id=i)
            text = exp.explanation_text
            for phrase in self.CANNED:
                assert phrase not in text, (decision, phrase)
            assert text.startswith(f"Credit Decision: {decision}\nCredit Score: {exp.predicted_score}/1000\n")
            assert f"Estimated Default Probability: {served[i]:.1%}" in text
            assert f"Baseline default probability {exp.base_value:.1%};" in text
            assert DECISION_REASONS[decision] in text
            # The listed factors ARE the attribution, in order, with their size.
            for marker, factors in (("✗", exp.top_negative_factors), ("✓", exp.top_positive_factors)):
                lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith(marker)]
                expected = [
                    f"{marker} {f['feature'].replace('_', ' ').title()} ({f['shap_value'] * 100:+.1f} pts)"
                    for f in factors[:3]
                ]
                assert lines == expected, (decision, marker)
            if decision == "Review":
                assert "no review workflow" in text


class TestAttributionIsFaithful:
    """Exact additivity is not faithfulness.

    Rescaling log-odds TreeSHAP by (change in served probability) / (change in margin)
    is exactly additive too, and that ratio explodes near the baseline. These tests check
    the attribution against an INDEPENDENT estimator of the same quantity — shap's
    KernelExplainer on the same served predict_proba and the same background, without L1
    regularisation (its default AIC selection zeroes small attributions) — on the rows
    nearest the baseline, where such a method fails, plus three others.

    Tolerances (measured 2026-11-04): the 32-permutation estimate is within
    0.46 pt of a converged 256-permutation run on the API model, and
    KernelSHAP at 4,096 samples carries error of its own; 1.5 points of
    probability covers both with margin. The largest gap measured here was
    0.66 pt (0.10-0.46 pt on the near-baseline rows), 2026-11-04.
    """

    @pytest.fixture(scope="class")
    def reference(self, fitted_explainer):
        import shap

        explainer, X_test = fitted_explainer
        inner = explainer._explainer
        served = explainer.model.predict_proba(X_test)[:, 1]
        near = list(np.argsort(np.abs(served - inner.expected_value))[:3])
        rows = near + [0, 1, 2]
        kernel = shap.KernelExplainer(inner._f, inner.background)
        refs = {i: _kernel_values(kernel, X_test.values[i:i + 1]) for i in rows}
        mine = {i: inner.shap_values(X_test.values[i:i + 1])[0] for i in rows}
        return rows, refs, mine

    def test_the_reference_does_not_depend_on_what_ran_before_it(self, fitted_explainer):
        """The two ambient seeds on which the unseeded reference failed this class."""
        import shap

        explainer, X_test = fitted_explainer
        inner = explainer._explainer
        kernel = shap.KernelExplainer(inner._f, inner.background)
        values = []
        for ambient in (15, 31):
            np.random.seed(ambient)
            values.append(_kernel_values(kernel, X_test.values[:1]))
            assert np.random.get_state()[1][0] == np.random.RandomState(ambient).get_state()[1][0]
        assert np.array_equal(values[0], values[1])

    def test_agrees_with_kernel_shap_in_sign_and_magnitude(self, reference):
        rows, refs, mine = reference
        for i in rows:
            ref, est = refs[i], mine[i]
            material = np.abs(ref) > 0.01
            assert material.any(), i
            assert (np.sign(est[material]) == np.sign(ref[material])).all(), i
            assert np.abs(est - ref).max() < 0.015, (i, np.abs(est - ref).max())

    def test_attribution_mass_is_not_inflated(self, reference):
        rows, refs, mine = reference
        for i in rows:
            assert np.abs(mine[i]).sum() <= 1.05 * np.abs(refs[i]).sum() + 0.005, i

    def test_attribution_mass_is_bounded_away_from_the_baseline(self, fitted_explainer):
        """Sanity bound over the whole test set. Near the baseline the ratio
        sum|phi| / |p - base| is unbounded for ANY faithful method (offsetting
        features); away from it (|p - base| >= 5 points) it must stay modest.
        Measured 2026-11-04 on this fixture model: median 1.8, p99 5.0, max
        5.8 (on the API model: median 2.3, p99 7.8, max 10.7). No all-rows bound is
        asserted: near the baseline the ratio is unbounded for any faithful
        method, so an all-rows comparison says nothing about faithfulness."""
        explainer, X_test = fitted_explainer
        inner = explainer._explainer
        served = explainer.model.predict_proba(X_test)[:, 1]
        phi = inner.shap_values(X_test.values, n_permutations=8)
        np.testing.assert_allclose(inner.expected_value + phi.sum(axis=1), served, atol=1e-9)
        delta = np.abs(served - inner.expected_value)
        far = delta >= 0.05
        assert far.sum() > 100
        assert (np.abs(phi[far]).sum(axis=1) / delta[far]).max() < 10


class TestAttributionOverInputs:
    """Every reason names something the applicant supplied, never a feature no applicant
    typed (monthly_payment_burden). The API attributes over its inputs, with the inputs it
    couples switching together, so a Shapley state cannot pair one applicant's income with
    another's deposits, a combination the API refuses.
    """

    @pytest.fixture(scope="class")
    def api_explainer(self):
        from creditlens.api import app as api

        api._startup_train()
        df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=api.DEFAULT_SEED))
        train, test = train_test_split(df, test_size=0.20, seed=api.DEFAULT_SEED)
        return api._model_state["explainer"], train, test

    def test_players_partition_the_inputs(self):
        from creditlens.features.engineering import ATTRIBUTION_GROUPS, MODEL_INPUTS, attribution_players

        players = attribution_players()
        fields = [f for group in players.values() for f in group]
        assert len(fields) == len(set(fields))
        assert set(fields) == set(MODEL_INPUTS)  # net cash flow is derived, never a player
        for name, group in ATTRIBUTION_GROUPS.items():
            assert players[name] == group

    def test_a_column_in_two_players_or_none_is_refused(self):
        from creditlens.explainability.shap_explainer import ServedShapleyExplainer

        background = np.zeros((2, 3))
        for groups in ([[0, 1], [1, 2]], [[0], [1]]):
            with pytest.raises(ValueError, match="exactly one player"):
                ServedShapleyExplainer(lambda a: a[:, 0], background, groups=groups)

    def test_every_reason_names_something_the_applicant_supplied(self):
        from fastapi.testclient import TestClient

        from creditlens.api.app import ApplicantFeatures, app

        applicant = {"annual_income": 30000, "loan_amount": 25000, "loan_term_months": 36,
                     "fico_score": 560, "debt_to_income": 0.62, "employment_years": 1,
                     "credit_utilization": 0.92, "credit_age_months": 18, "num_hard_inquiries": 6,
                     "num_delinquencies": 2, "months_since_delinquency": 5,
                     "monthly_inflow": 2500, "monthly_outflow": 2400, "avg_account_balance": 1000,
                     "paycheck_regularity": 0.5, "bill_payment_consistency": 0.5, "nsf_count": 3,
                     "overdraft_days": 8, "rent_payment_score": 0.5, "utility_payment_score": 0.5,
                     "banking_tenure_months": 6, "savings_rate": 0.0}
        with TestClient(app) as c:
            body = c.post("/explain/1", json=applicant).json()
        players = body["attribution_players"]
        assert set(body["shap_values"]) == set(players)
        for fields in players.values():
            assert set(fields) <= set(ApplicantFeatures.model_fields)
            assert set(fields) <= set(applicant)  # the caller sent each of them
        reasons = {f["feature"] for f in body["top_negative_factors"] + body["top_positive_factors"]}
        assert reasons and reasons <= set(players)
        for derived in ("monthly_payment_burden", "monthly_net_cashflow"):
            assert derived not in body["shap_values"]
        assert body["base_value"] + sum(body["shap_values"].values()) == pytest.approx(
            body["default_probability"], abs=5.1e-5)

    def test_every_state_evaluated_is_an_applicant_the_api_accepts(self, api_explainer):
        """Read the states the value function really receives and put each
        through the API's own validators, joint rules included."""
        from creditlens.api.app import ApplicantFeatures
        from creditlens.data.domain import INTEGER_FIELDS
        from creditlens.explainability.shap_explainer import ServedShapleyExplainer

        explainer, _, test = api_explainer
        inner = explainer._explainer
        states = []
        # A slice of the background: every state still pairs real rows, and each
        # one goes through pydantic.
        probe = ServedShapleyExplainer(
            lambda a: states.append(a.copy()) or np.zeros(len(a)), inner.background[::20], groups=inner.groups)
        states.clear()
        delinquent = test.index[test["num_delinquencies"] > 0][0]
        probe.shap_values(explainer._columns(test.loc[[0, delinquent]]), n_permutations=8)
        evaluated = np.unique(np.vstack(states), axis=0)
        assert len(evaluated) > 2000  # non-vacuous
        columns = explainer.input_fields
        ints = [c in INTEGER_FIELDS for c in columns]
        for row in evaluated:
            ApplicantFeatures(**{c: int(v) if i else float(v) for c, v, i in zip(columns, row, ints)})

    def test_the_payment_rule_holds_for_any_recombination_of_training_values(self, api_explainer):
        """Loan amount, term and income are separate players, so a state can
        pair them across applicants. Inside the training range that can never
        break the rule; only an applicant in the domain's 5% margin can (module
        docstring), and the response's warnings already name that input."""
        from creditlens.data.domain import MAX_PAYMENT_BURDEN, VALIDITY_DOMAIN

        _, train, _ = api_explainer
        worst = (train["loan_amount"].max() / train["loan_term_months"].min()) / (train["annual_income"].min() / 12)
        assert worst == pytest.approx(2.67, abs=0.01)
        assert worst < MAX_PAYMENT_BURDEN
        margin = (VALIDITY_DOMAIN["loan_amount"][1] / train["loan_term_months"].min()) / (train["annual_income"].min() / 12)
        assert margin > MAX_PAYMENT_BURDEN

    def test_agrees_with_kernel_shap_over_the_same_players(self, api_explainer):
        """The independent check, on the API's model and players: shap's
        KernelExplainer, same function and groups, against a slice of the background
        (its cost grows with background rows, and the estimator is the same at any
        size), no L1. Measured on the same slice
        2026-11-04: largest gap 0.44 pt on 12 applicants (8,192 samples)."""
        import shap
        from shap.utils._legacy import DenseData

        from creditlens.decision import probability_to_score

        from creditlens.explainability.shap_explainer import ServedShapleyExplainer

        explainer, _, test = api_explainer
        full = explainer._explainer
        inner = ServedShapleyExplainer(full._f, full.background[::20], groups=full.groups)
        X = explainer._columns(test)
        served = np.asarray(inner._f(X), dtype=float)
        near = list(np.argsort(np.abs(served - inner.expected_value))[:3])
        denied = list(np.where(probability_to_score(served) < 600)[0][:3])
        kernel = shap.KernelExplainer(
            inner._f, DenseData(inner.background, list(explainer.players), inner.groups))
        for i in near + denied + [0, 1, 2]:
            ref = _kernel_values(kernel, X[i:i + 1])
            est = inner.shap_values(X[i:i + 1])[0]
            assert len(ref) == len(est) == len(explainer.players)
            material = np.abs(ref) > 0.01
            assert material.any(), i
            assert (np.sign(est[material]) == np.sign(ref[material])).all(), i
            assert np.abs(est - ref).max() < 0.015, (i, np.abs(est - ref).max())
            assert np.abs(est).sum() <= 1.05 * np.abs(ref).sum() + 0.005, i


def test_the_served_baseline_is_the_populations_default_rate():
    """/explain's base_value sits at the 14.9% the training data defaults at, so every
    reason is relative to the right number."""
    from fastapi.testclient import TestClient

    from creditlens.api import app as api

    df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=api.DEFAULT_SEED))
    train, _ = train_test_split(df, test_size=0.20, seed=api.DEFAULT_SEED)
    applicant = {"annual_income": 75000, "loan_amount": 12000, "loan_term_months": 36, "debt_to_income": 0.32,
                 "employment_years": 4, "fico_score": 720, "credit_utilization": 0.25, "num_delinquencies": 0,
                 "months_since_delinquency": -1, "num_hard_inquiries": 1, "credit_age_months": 84,
                 "monthly_inflow": 6250, "monthly_outflow": 4800, "paycheck_regularity": 0.92,
                 "bill_payment_consistency": 0.88, "nsf_count": 0, "overdraft_days": 0, "rent_payment_score": 0.95,
                 "utility_payment_score": 0.85, "banking_tenure_months": 48, "avg_account_balance": 8500,
                 "savings_rate": 0.1}
    with TestClient(api.app) as client:
        api._startup_train()
        base = client.post("/explain/0", json=applicant).json()["base_value"]
    assert abs(base - train["default"].mean()) < 0.002, (base, train["default"].mean())


class TestReasonsDoNotDependOnTheBackgroundDraw:
    """The background is 1,000 rows, every k-th by served probability. This redraws it by
    shifting where that stride starts, and README publishes what moves."""

    @pytest.fixture(scope="class")
    def redraws(self):
        from creditlens.api import app as api
        from creditlens.decision import probability_to_score
        from creditlens.explainability.shap_explainer import ServedShapleyExplainer, background_sample

        api._startup_train()
        df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=api.DEFAULT_SEED))
        train, test = train_test_split(df, test_size=0.20, seed=api.DEFAULT_SEED)
        explainer = api._model_state["explainer"]
        f, groups = explainer._explainer._f, explainer._explainer.groups
        rows, X = explainer._columns(train), explainer._columns(test)
        p = np.asarray(f(rows), dtype=float)
        not_approved = X[probability_to_score(f(X)) < 600][:20]
        bases, phis = [], []
        for offset in np.linspace(0.05, 0.95, 10):
            inner = ServedShapleyExplainer(f, rows[background_sample(p, 1000, offset)], groups=groups)
            bases.append(inner.expected_value)
            phis.append(inner.shap_values(not_approved))
        return np.array(bases), np.array(phis), float(p.mean())

    def test_the_baseline_is_the_population_mean(self, redraws):
        bases, _, mean = redraws
        assert np.abs(bases - mean).max() < 0.002, (bases, mean)

    def test_the_readme_states_the_remaining_variation(self, redraws):
        import re
        from pathlib import Path

        bases, phis, _ = redraws
        gap = (phis.max(axis=0) - phis.min(axis=0)).max() * 100

        def changed(k: int) -> float:
            per_row = [{frozenset(np.argsort(-phis[r, i])[:k]) for r in range(len(phis))} for i in range(phis.shape[1])]
            return float(np.mean([len(s) > 1 for s in per_row]))

        assert gap < 3.0
        assert changed(3) < 0.29 and changed(4) < 0.39
        text = re.sub(r"\s+", " ", (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8"))
        assert (f"the baseline stays {bases.min() * 100:.2f}% to {bases.max() * 100:.2f}%, no factor moves by more than "
                f"{gap:.2f} points, and the top 3 reasons change for {changed(3):.0%} of the not-approved, the top 4 "
                f"for {changed(4):.0%}") in text
