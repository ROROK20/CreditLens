"""Claims this project must not make.

Each pattern below is a sentence that would be false here. The test reads every
shipped text surface (docs, Python source, frontend source) and fails if one
appears.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SHIPPED = sorted(
    [ROOT / "README.md", ROOT / "SECURITY.md", ROOT / "CONTRIBUTING.md", ROOT / "CHANGELOG.md",
     ROOT / "frontend" / "README.md"]
    + list((ROOT / "docs").glob("*.md"))
    + list((ROOT / "creditlens").rglob("*.py"))
    + list((ROOT / "frontend" / "app").rglob("*.tsx"))
    + list((ROOT / "frontend" / "lib").rglob("*.ts"))
    + list((ROOT / "frontend" / "components").rglob("*.tsx"))
)

UNSUPPORTED_CLAIMS = {
    r"legal standard under (?:US )?EEOC":
        "the EEOC four-fifths rule governs employment selection, not lending",
    r"compliance metrics":
        "no metric here is a legal compliance test",
    r"required for fair lending compliance":
        "neither ECOA nor the FHA prescribes these metrics or thresholds",
    r"credit officer will contact":
        "there is no review workflow; nothing contacts anyone",
    r"emergency fund|Reduce credit utilization below":
        "canned advice not derived from the applicant's attribution",
    r"fixes all three":
        "a synthetic-data reference implementation fixes nothing in production lending",
    r"APR, risk tier, (?:SHAP )?waterfall":
        "APR and risk tier are a client-side score-band lookup, not engine outputs",
    r"Estimated APR":
        "the APR is an illustrative score-band lookup, not an estimate",
    r"Strong credit profile":
        "decision reasons must state the band rule, true for every applicant",
    r'45M Americans are "credit invisible"':
        "CFPB 2015: 26M credit invisible + 19M unscorable",
    r"\b[Rr]ealistic\b|\b[Rr]ealism\b":
        "the generator's features are mostly independent draws; tests/test_generator.py measures it",
    r"powered by a real Python engine":
        "a build with no backend runs recorded weights in the browser",
}


def _normalized(path: Path) -> str:
    """Whitespace collapsed and Markdown quote markers dropped, so a claim
    wrapped across lines or inside a > blockquote still matches."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    return re.sub(r"\s+", " ", text)


def test_shipped_surfaces_are_found():
    names = {p.name for p in SHIPPED}
    assert {"README.md", "usage.md", "architecture.md", "metrics.py", "page.tsx", "scoring.ts"} <= names


@pytest.mark.parametrize("pattern", sorted(UNSUPPORTED_CLAIMS), ids=lambda p: p[:32])
def test_banned_claim_is_absent(pattern):
    hits = [p.relative_to(ROOT).as_posix() for p in SHIPPED if re.search(pattern, _normalized(p))]
    assert not hits, f"{UNSUPPORTED_CLAIMS[pattern]} — found in {hits}"


def test_frontend_categorises_and_labels_real_model_features():
    """The live waterfall tags each SHAP row core / behavioral / alternative. A row
    is an attribution player (an input, or a group of inputs that move together):
    every player needs a category and a label, and every key must be a player."""
    from creditlens.features.engineering import attribution_players

    features = set(attribution_players())
    ts = (ROOT / "frontend" / "lib" / "scoring.ts").read_text(encoding="utf-8")

    def keys(const: str) -> set[str]:
        block = ts[ts.index(f"const {const}"):]
        block = block[block.index("{", block.index("=")):block.index("};")]
        return set(re.findall(r"(\w+):", block))

    categories = keys("API_FEATURE_CATEGORY")
    assert features <= categories, sorted(features - categories)
    assert categories == features, sorted(categories ^ features)
    assert keys("API_FEATURE_LABELS") == features, sorted(keys("API_FEATURE_LABELS") ^ features)
    assert 'category: apiFeatureCategory(f.feature)' in (ROOT / "frontend" / "app" / "page.tsx").read_text()


def test_ensembling_and_beyond_fico_claims_are_gone():
    """fico_score is a required input and FICO is the top risk reason in most denials,
    so no text may say the engine "goes beyond FICO"; no ensembling code exists, so none
    may offer the MLP "for ensembling"."""
    for pattern, why in (
        (r"[Gg]oes beyond FICO", "fico_score is a required input"),
        (r"for ensembl|available for ensembling", "no ensembling code exists"),
    ):
        hits = [p.relative_to(ROOT).as_posix() for p in SHIPPED if re.search(pattern, _normalized(p))]
        assert not hits, f"{why} — found in {hits}"
    readme = _normalized(ROOT / "README.md")
    assert "This demo requires a FICO score" in readme


def test_documented_no_effect_fields_match_the_ground_truth():
    """The README names the generated fields that have no effect on the
    synthetic default label (and so are not model inputs). Check the list
    against the ground truth, data/generator.py default_log_odds(), and check
    that every other model input does have an effect."""
    from creditlens.api.app import ApplicantFeatures
    from creditlens.data.generator import LABEL_EFFECTS
    from creditlens.features.engineering import MODEL_INPUTS

    text = _normalized(ROOT / "README.md")
    listed = text[text.index("Four generated fields have no effect on it"):text.index("FICO carries the strongest")]
    no_effect = set(re.findall(r"`(\w+)`", listed))
    assert no_effect == {"num_open_accounts", "num_subscriptions", "loan_purpose", "interest_rate"}
    model_inputs = set(MODEL_INPUTS)
    assert model_inputs <= set(ApplicantFeatures.model_fields)
    assert not no_effect & model_inputs
    assert not no_effect & set(ApplicantFeatures.model_fields)
    assert set(LABEL_EFFECTS) == model_inputs


def test_the_social_card_holds_for_a_build_with_no_engine():
    """The card must hold for every build, including one that runs recorded log-odds in
    the browser, so it names neither SHAP nor an engine."""
    layout = (ROOT / "frontend" / "app" / "layout.tsx").read_text(encoding="utf-8")
    description = re.search(r'const description =\s*"([^"]+)"', layout).group(1)
    assert "SHAP" not in description and "engine" not in description
    assert "synthetic data" in description
    assert layout.count("description,") == 3 and "description:" not in layout


def test_the_architecture_doc_counts_what_the_code_has():
    from creditlens.features.engineering import MODEL_INPUTS, attribution_players

    players = attribution_players()
    alone = sum(len(fields) == 1 for fields in players.values())
    text = _normalized(ROOT / "docs" / "architecture.md")
    assert f"{len(MODEL_INPUTS)} of them are model inputs" in text
    assert f"Attribution is over {len(players)} factors: {alone} inputs alone" in text
    assert "largest training burden" not in text


def test_stale_wording_stays_gone():
    app = (ROOT / "creditlens" / "api" / "app.py").read_text(encoding="utf-8")
    assert "Three joint" not in app and "Five joint" in app
    assert "410 / Denied" not in app and "421 / Denied" in app
    assert "badge/tests-" not in (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_audit_states_its_multiplicity_and_interval_limits():
    """The overall verdict reads FAIL if any of four criteria fails, with no adjustment
    across them, and the equalized-odds interval adds the bounds of its two signed gaps,
    which is conservative. Both are stated, in the README and in the module that computes
    them."""
    from creditlens.fairness import metrics

    for text in (_normalized(ROOT / "README.md"), re.sub(r"\s+", " ", metrics.__doc__)):
        assert "up to 20% of the time" in text
        assert "both signed intervals span 0" in text and "onservative" in text and "absolute gap" in text


def test_no_file_carries_a_date_later_than_today():
    """A date stamp is a claim about when something was measured or captured, so no
    file may carry a date after the day the tests run."""
    import datetime

    today = datetime.date.today()
    skip_dirs = {".git", "node_modules", ".next", "__pycache__", ".pytest_cache"}
    suffixes = {".py", ".md", ".ts", ".tsx", ".mjs", ".js", ".txt", ".yml", ".yaml", ".toml", ".cfg"}
    future = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in suffixes and not skip_dirs & set(path.parts):
            for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                for y, m, d in re.findall(r"\b(20[0-9]{2})-([01][0-9])-([0-3][0-9])\b", line):
                    try:
                        stamp = datetime.date(int(y), int(m), int(d))
                    except ValueError:
                        continue
                    if stamp > today:
                        future.append(f"{path.relative_to(ROOT)}:{n} {stamp}")
    assert not future, future


def test_the_no_backend_model_names_what_the_engine_names():
    """The browser fallback runs the engine's recorded model
    (tests/test_engine_snapshot.py), and every factor it draws is one the live chart
    names."""
    engine_model = (ROOT / "frontend" / "lib" / "engineModel.mjs").read_text(encoding="utf-8")
    ts = (ROOT / "frontend" / "lib" / "scoring.ts").read_text(encoding="utf-8")
    factors = engine_model[engine_model.index("const FACTORS"):engine_model.index("/** Each factor's change")]
    keys = set(re.findall(r"(\w+): \[", factors))
    assert len(keys) >= 10
    labels = ts[ts.index("const API_FEATURE_LABELS"):ts.index("const API_FEATURE_CATEGORY")]
    assert keys <= set(re.findall(r"(\w+): \"", labels)), sorted(keys)
    assert "Loan-to-Income" not in ts and "Loan-to-Income" not in engine_model


def test_production_builds_do_not_fall_back_to_localhost():
    """The URL fallbacks are next-dev-only (frontend/lib/apiUrl.mjs, siteUrl.mjs;
    `npm test` covers their logic). This pins the wiring: every caller uses them,
    the engine client refuses to make a call without one, relative preview URLs
    are omitted with the base (Next resolves them against localhost otherwise),
    and the build runs the prerendered-page check."""
    frontend = ROOT / "frontend"
    api = (frontend / "lib" / "api.ts").read_text(encoding="utf-8")
    assert "resolveApiUrl(process.env.NEXT_PUBLIC_API_URL, process.env.NODE_ENV)" in api
    assert 'if (API_URL === null) throw new Error("No engine configured' in api

    layout = (frontend / "app" / "layout.tsx").read_text(encoding="utf-8")
    assert "resolveSiteUrl(process.env.NEXT_PUBLIC_SITE_URL, process.env.NODE_ENV)" in layout
    assert "...(SITE_URL ? { metadataBase: new URL(SITE_URL) } : {})" in layout
    assert '...(SITE_URL ? { url: "/" } : {})' in layout
    assert "images: OG_IMAGES" in layout and "http://localhost" not in layout

    package = json.loads((frontend / "package.json").read_text(encoding="utf-8"))
    assert package["scripts"]["postbuild"] == "node scripts/check-static.mjs"
    assert "lib/apiUrl.test.mjs" in package["scripts"]["test"]
    assert "lib/siteUrl.test.mjs" in package["scripts"]["test"]


def test_the_docs_do_not_present_train_as_the_way_to_train():
    """The API trains at startup, and POST /train answers 403 unless the operator opts in.
    Anything that mentions it as a way to train also says how it is enabled."""
    for path in (ROOT / "CONTRIBUTING.md", ROOT / "docs" / "architecture.md", ROOT / "docs" / "usage.md",
                 ROOT / "SECURITY.md"):
        assert "`POST /train`" in path.read_text(encoding="utf-8"), path.name
        assert "CREDITLENS_ENABLE_TRAIN=1" in path.read_text(encoding="utf-8"), path.name
    assert "via `POST /train`" not in _normalized(ROOT / "docs" / "architecture.md")
    assert "trigger it explicitly" not in _normalized(ROOT / "CONTRIBUTING.md")


def test_one_sentence_describes_the_project_everywhere():
    """setup.py, the frontend's package.json and the OpenAPI metadata carry the README's
    first sentence."""
    from creditlens.api.app import app

    first = re.search(r"^\*\*(.+?)\*\*$", (ROOT / "README.md").read_text(encoding="utf-8"), re.M).group(1)
    assert first.startswith("An explainable credit-scoring reference:")
    assert f'description="{first}"' in (ROOT / "setup.py").read_text(encoding="utf-8")
    assert json.loads((ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))["description"] == first
    assert app.description == first


def test_shipped_text_says_what_is_not_what_was():
    """Comments, docstrings and docs state what the code does and why. History belongs in
    the commit log, and a date belongs only on a measurement."""
    history = re.compile(r"\b(?:[Uu]ntil|[Bb]efore|[Ss]ince) 20\d\d-\d\d|\bno longer\b|\bpreviously\b|\bold code\b"
                         r"|\bwas wrong\b|\bof the day\b|\bused to (?:be|read|say|score|default|fall|drop|show|leave)\b")
    surfaces = SHIPPED + [ROOT / "requirements.txt", ROOT / "setup.py", ROOT / "e2e" / "smoke.mjs",
                          ROOT / "e2e" / "capture.mjs", ROOT / "e2e" / "offline.mjs"]
    hits = [f"{p.relative_to(ROOT)}: {m.group(0)}" for p in surfaces for m in history.finditer(_normalized(p))]
    assert not hits, hits


def test_ci_runs_with_a_read_only_token():
    """A public repo takes pull requests from strangers, and CI needs nothing more."""
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        assert re.search(r"^permissions:\n  contents: read$", text, re.M), path.name
        assert "pull_request_target" not in text, path.name


def test_the_public_urls_are_documented_as_build_time():
    """NEXT_PUBLIC_* is inlined when the bundle is built, so the Dockerfile takes it as a
    build arg, compose passes it as one (at runtime it does nothing), and the README
    states no localhost default for every build."""
    docker = (ROOT / "frontend" / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^ARG NEXT_PUBLIC_API_URL$", docker, re.M)
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "environment:" not in compose and "args:" in compose
    readme = _normalized(ROOT / "README.md")
    assert "(default `http://localhost:8002`)" not in readme
    assert "Next inlines that variable at **build** time" in readme
    assert "a production build is demo-only: it makes no engine call at all" in readme
