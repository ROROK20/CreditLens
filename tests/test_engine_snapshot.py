"""What a visitor sees when no engine is connected.

A frontend-only deploy has no backend, so the page's fallback IS the product page. It
runs the engine's own served model from recorded weights, and the prerendered HTML ships
no number. These tests hold that recording to the live engine;
`npm test` (lib/engineModel.test.mjs) holds the browser code to the recording, and
`npm run build` fails if the prerendered page ships a score.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from creditlens.api import app as api
from creditlens.data.generator import DatasetConfig, generate_credit_dataset, train_test_split
from creditlens.decision import APPROVE_MIN_SCORE, REVIEW_MIN_SCORE, RISK_BANDS, probability_to_score, score_band_table

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = json.loads((ROOT / "frontend" / "lib" / "engineSnapshot.json").read_text(encoding="utf-8"))
PAGE = (ROOT / "frontend" / "app" / "page.tsx").read_text(encoding="utf-8")
SCORING = (ROOT / "frontend" / "lib" / "scoring.ts").read_text(encoding="utf-8")
ENGINE_MODEL = (ROOT / "frontend" / "lib" / "engineModel.mjs").read_text(encoding="utf-8")


def _recorder():
    spec = importlib.util.spec_from_file_location("record_engine", ROOT / "frontend" / "scripts" / "record_engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        api._startup_train()
        yield c


@pytest.fixture(scope="module")
def recorder():
    return _recorder()


def test_the_recorded_weights_are_the_served_models(client, recorder):
    live = recorder.model_snapshot()
    assert SNAPSHOT["model"]["intercept"] == pytest.approx(live["intercept"])
    assert SNAPSHOT["model"]["weights"] == pytest.approx(live["weights"])
    assert SNAPSHOT["model"]["feature_means"] == pytest.approx(live["feature_means"])
    assert SNAPSHOT["rules"] == recorder.rules_snapshot()
    assert SNAPSHOT["seed"] == api.DEFAULT_SEED and SNAPSHOT["n_samples"] == api.DEFAULT_TRAIN_SAMPLES


def test_every_recorded_answer_is_the_engines(client):
    vectors = SNAPSHOT["vectors"]
    assert sum(v["status"] == 200 for v in vectors) >= 8 and sum(v["status"] == 422 for v in vectors) >= 5
    for v in vectors:
        res = client.post("/score", json=v["applicant"])
        assert res.status_code == v["status"], (v["name"], res.text[:200])
        if v["status"] == 200:
            body = res.json()
            assert (body["credit_score"], body["default_probability"], body["decision"]) == (
                v["credit_score"], v["default_probability"], v["decision"]), v["name"]


def test_every_recorded_warning_is_the_engines(client):
    """The no-backend page shows these, so they must be the engine's."""
    scored = [v for v in SNAPSHOT["vectors"] if v["status"] == 200]
    assert sum(bool(v["warnings"]) for v in scored) >= 5
    for v in scored:
        assert client.post("/score", json=v["applicant"]).json()["warnings"] == v["warnings"], v["name"]


def test_the_recording_covers_what_only_a_field_bound_refuses(client):
    """Balance $465 on $15,000 of income: the API refused, the page scored 983."""
    by_name = {v["name"]: v for v in SNAPSHOT["vectors"]}
    for name in ("balance under its floor at low income", "FICO under its floor", "delinquencies over their ceiling",
                 "recency without a delinquency", "a delinquency without a recency", "a fractional count"):
        assert by_name[name]["status"] == 422, name
    assert "balance_warning" in SNAPSHOT["rules"] and "fields" in SNAPSHOT["rules"]


class TestThePageSaysOneThingAtATime:
    """One failed /explain must not strand the page on demo, and the note and the audit's
    badge must agree. e2e/smoke.mjs drives this in a browser; these pin the wiring."""

    def test_the_audit_is_live_only_while_the_engine_is(self):
        assert 'const fairnessIsLive = mode === "live" && apiFairness !== null;' in PAGE
        assert "fairnessIsLive ? apiFairness : recordedFairnessReport()" in PAGE

    def test_a_lost_engine_is_probed_until_it_answers(self):
        assert "setInterval(probe, RECONNECT_MS)" in PAGE
        assert 'API_URL === null ? undefined : setInterval' in PAGE  # a no-backend build makes no call
        failure = PAGE[PAGE.index('setMode("demo");'):PAGE.index("}, [applicant, mode]);")]
        for reset in ("setApiResult(null)", "setApiFairness(null)", "setRejection(null)", "setApiWarnings([])"):
            assert reset in failure, reset

    def test_the_no_backend_page_shows_warnings(self):
        assert 'mode === "demo" ? engineWarnings(toApiApplicant(applicant))' in PAGE
        assert "{warnings.length > 0 && (" in PAGE
        assert "export function engineWarnings" in ENGINE_MODEL


def test_the_recorded_audit_is_the_engines(client, recorder):
    report = client.get("/fairness/report").json()
    assert SNAPSHOT["fairness"] == recorder.fairness_snapshot(report)
    assert SNAPSHOT["score_bands"] == report["score_bands"]


def test_a_recorded_weight_scores_like_the_engine_in_python_too(client):
    """The formula the browser runs, checked here as well, on the audit applicants."""
    model = SNAPSHOT["model"]
    for v in (x for x in SNAPSHOT["vectors"] if x["status"] == 200):
        a = v["applicant"]
        months = a["months_since_delinquency"]
        x = {**a, "months_since_delinquency": min(months, 24) if months >= 0 else 24,
             "monthly_net_cashflow": a["monthly_inflow"] - a["monthly_outflow"],
             "avg_account_balance": np.log1p(a["avg_account_balance"] / 1000),
             "monthly_payment_burden": (a["loan_amount"] / a["loan_term_months"]) / (a["annual_income"] / 12)}
        log_odds = model["intercept"] + sum(w * x[name] for name, w in model["weights"].items())
        assert probability_to_score(1 / (1 + np.exp(-log_odds))) == v["credit_score"], v["name"]


def test_the_fallback_has_no_hand_set_weights_and_says_what_it_is():
    assert "logOdds +=" not in SCORING and "hand-set" not in SCORING
    note = re.sub(r"\s+", " ", PAGE[PAGE.index('data-testid="demo-note"'):])
    note = note[:note.index("</p>")]
    assert "the engine&apos;s recorded model" in note and "not the live engine&apos;s Shapley values" in note
    assert 'mode === "demo" && (' in PAGE


class TestNoNumberBeforeTheEngineAnswers:
    def test_the_page_has_a_pending_state_and_no_local_number_under_the_live_badge(self):
        assert 'data-testid="score-pending"' in PAGE
        assert 'const result: ScoringResult | null = mode === "live" ? apiResult : recorded;' in PAGE
        assert 'mode === "demo" ? scoreApplicant(applicant) : null' in PAGE
        assert "mockResult" not in PAGE

    def test_no_recorded_audit_under_the_live_badge(self):
        """While the first report is pending no recording shows under "Live engine".
        e2e/smoke.mjs drives this in a browser."""
        assert 'const fairnessPending = mode === "live" && apiFairness === null;' in PAGE
        assert 'const showVerdicts = mode !== "checking" && !fairnessPending;' in PAGE
        assert 'data-testid="audit-pending"' in PAGE
        assert PAGE.index("{showVerdicts && (<>") > PAGE.index('data-testid="audit-pending"')
        assert "setTimeout(() => setFairnessAttempt((n) => n + 1), RECONNECT_MS)" in PAGE

    def test_the_build_fails_if_the_prerender_ships_a_score(self):
        script = (ROOT / "frontend" / "scripts" / "check-static.mjs").read_text(encoding="utf-8")
        for needle in ('score-value', 'score-pending', 'fairness-verdict'):
            assert needle in script
        package = json.loads((ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
        assert package["scripts"]["postbuild"] == "node scripts/check-static.mjs"
        assert "lib/engineModel.test.mjs" in package["scripts"]["test"]


class TestRiskBandsAreMeasured:
    """"Prime" covered 81% of applicants, its 750-799 band observed 16.9% default (above
    the 14.9% base rate) at a 6.5% APR, and Near-Prime observed 34.8% at 9.5%."""

    def _table(self, seed: int) -> list[dict]:
        from creditlens.features.engineering import FeatureEngineer
        from creditlens.models.scorer import LogisticScorer

        df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=seed))
        train, test = train_test_split(df, test_size=0.20, seed=seed)
        engineer = FeatureEngineer(scale=True)
        X, y = engineer.fit_transform(train)
        model = LogisticScorer().fit(X, y)
        p = model.predict_proba(engineer.transform(test)[0])[:, 1]
        return score_band_table(probability_to_score(p), test["default"].to_numpy())

    @pytest.mark.parametrize("seed", [42, 7, 123])
    def test_bands_order_by_observed_default_rate(self, seed):
        rates = [row["observed_default_rate"] for row in self._table(seed)]
        assert rates == sorted(rates), (seed, rates)
        assert len(set(rates)) == len(rates), (seed, rates)

    def test_bands_nest_inside_the_decision_cutoffs(self):
        cutoffs = [lower for lower, _ in RISK_BANDS]
        assert APPROVE_MIN_SCORE in cutoffs and REVIEW_MIN_SCORE in cutoffs and cutoffs[-1] == 0

    def test_the_lowest_band_above_the_base_rate_is_named_for_it(self, client):
        """A band may not read safe while observing more than the base rate."""
        table = client.get("/fairness/report").json()["score_bands"]
        by_label = {row["label"]: row["observed_default_rate"] for row in table}
        assert by_label["Very low"] < by_label["Low"] < 0.149 < by_label["Elevated"]

    def test_the_readme_publishes_the_table(self, client):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        rows = re.findall(r"^\| (Very low|Low|Elevated|High|Very high) \| (\d+)–(\d+) \| (\d+) \| ([\d.]+)% \| ([\d.]+)% \|$",
                          readme, re.M)
        table = client.get("/fairness/report").json()["score_bands"]
        assert [(r[0], int(r[1]), int(r[2]), int(r[3]), float(r[4]), float(r[5])) for r in rows] == [
            (b["label"], b["min_score"], b["max_score"], b["n"], round(b["share"] * 100, 1),
             round(b["observed_default_rate"] * 100, 1)) for b in table]

    def test_there_is_no_apr_or_tier_left_to_contradict_it(self):
        for text in (PAGE, SCORING):
            for word in ("APR", "Prime", "Subprime", "scoreToRiskTier"):
                assert word not in text, word


class TestPresetsAreWhatTheySay:
    def _percentile(self, client, score: int) -> float:
        df = generate_credit_dataset(DatasetConfig(n_samples=api.DEFAULT_TRAIN_SAMPLES, seed=api.DEFAULT_SEED))
        _, test = train_test_split(df, test_size=0.20, seed=api.DEFAULT_SEED)
        model, engineer = api._model_state["model"], api._model_state["engineer"]
        scores = probability_to_score(model.predict_proba(engineer.transform(test)[0])[:, 1])
        return float((scores < score).mean() * 100)

    def test_labels_match_where_the_presets_sit(self, client):
        by_name = {v["name"]: v for v in SNAPSHOT["vectors"]}
        assert self._percentile(client, by_name["strong"]["credit_score"]) >= 90
        assert 35 <= self._percentile(client, by_name["median"]["credit_score"]) <= 65
        assert self._percentile(client, by_name["weak"]["credit_score"]) <= 10
        assert ">Median Applicant</button>" in PAGE and ">Average</button>" not in PAGE

    def test_the_readme_places_the_default_applicant(self, client):
        score = next(v["credit_score"] for v in SNAPSHOT["vectors"] if v["name"] == "default")
        pct = self._percentile(client, score)
        readme = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
        assert f"page's default applicant sits at the {pct:.0f}th percentile" in readme

    def test_the_page_presets_are_the_recorded_applicants(self):
        names = {camel: snake for snake, camel in re.findall(
            r"(\w+): (?:.*\? )?a\.(\w+)",
            (ROOT / "frontend" / "lib" / "api.ts").read_text(encoding="utf-8").split("function toApiApplicant")[1])}
        base_block = SCORING[SCORING.index("export const DEFAULT_APPLICANT"):]
        base_block = base_block[:base_block.index("};")]
        base = {k: float(v.replace("_", "")) for k, v in re.findall(r"(\w+): ([\d_.]+),", base_block)}
        by_name = {v["name"]: v for v in SNAPSHOT["vectors"]}
        presets = dict(re.findall(r'preset === "(\w+)"\) setApplicant\(\{(.*?)\}\);', PAGE, re.S))
        assert set(presets) == {"strong", "median", "weak"}
        for name, body in presets.items():
            merged = {**base, **{k: float(v) for k, v in re.findall(r"(\w+): ([\d.]+)", body)}}
            wire = {names[k]: v for k, v in merged.items()}
            if wire["num_delinquencies"] == 0:
                wire["months_since_delinquency"] = -1
            assert wire == {k: float(v) for k, v in by_name[name]["applicant"].items()}, name
