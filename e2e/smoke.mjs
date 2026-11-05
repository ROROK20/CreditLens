/**
 * CreditLens scripted live-fire smoke driver: the end-to-end check that the
 * documented demo reproduces.
 *
 * Proves in ONE free, repeatable command that CreditLens's deterministic
 * (seed=42) scoring + honest fairness audit — its signature —
 * reproduce over HTTP AND in a real browser against the real Python engine
 * (FastAPI + logistic scorer + SHAP), with zero console errors.
 *
 * The signature (deterministic under seed=42 AND the exact dependency pins in
 * requirements.txt; re-derived in-process 2026-11-04):
 *   - the default applicant scores exactly EXPECT_SCORE below ("Approved"), the
 *     demo's headline score, and its default_probability is the score's own
 *     probability (score == round((1 - p) * 1000));
 *   - the fairness audit is computed on the FULL test set across ALL three
 *     demographic groups (no cherry-picked slices: auditing a subset changes
 *     what the criteria mean):
 *     groups A/B/C carry n_samples 1234/491/275 (sum 2000 = 20% of the
 *     10,000-row synthetic set); group A = the "1234". The split depends only on the
 *     seed, the training size and the group draw;
 *   - the audit is interval-aware: disparate impact PASSES its
 *     0.80 screen on its simultaneous 95% interval (lower bound 0.9149, measured
 *     2026-11-04), every criterion carries an interval and a
 *     PASS/FAIL/INCONCLUSIVE verdict, and the overall verdict follows from them
 *     (INCONCLUSIVE on this audit set; README "What the demo reports, and why").
 *     Demographic parity reads PASS on this model (CI 0–0.0778), but a verdict that
 *     moves with the model is not a signature, so it is not asserted.
 *
 * How exact the score is. The number is derived from code and exact for THIS payload
 * only; an input is never chosen because it reproduces a number. Neighbourhood swept
 * in-process (2026-11-04): FICO
 * 710–715 → 992, 718–722 → 993, 725–730 → 994; debt_to_income 0.31–0.34 → 993
 * (0.30 → 994); credit_utilization 0.24–0.26 → 993 (0.20 → 994, 0.30 → 992);
 * annual_income 70k–80k → 993; loan_amount 10k–14k → 993; bill_payment
 * 0.86–0.90 → 993; savings_rate 0.08–0.12 → 993. A smooth curve, with no split
 * to sit on: the neighbours differ by the rounding of one point.
 *
 * It self-provisions: spawns the FastAPI engine (:8002) and the Next.js UI
 * (:3003), waits for readiness (incl. startup model training), asserts, and
 * tears BOTH down in a `finally` (also on failure). Nothing is left running.
 *
 *   node e2e/smoke.mjs        # from the repo root; exit 0 = all pass, 1 = any fail
 *
 * Overrides (defaults unchanged): CREDITLENS_API_PORT runs the engine on
 * another port when 8002 is taken by something else — the UI is pointed at it
 * through NEXT_PUBLIC_API_URL; CREDITLENS_PYTHON picks the interpreter that has
 * the pinned requirements (default `python3`).
 *   CREDITLENS_API_PORT=8012 CREDITLENS_PYTHON=python3.12 node e2e/smoke.mjs
 *
 * Drives the Playwright *library* from a plain .mjs — NOT @playwright/test
 * (its TS spec loader breaks under Node 23). Playwright is a devDependency of
 * frontend/, so it is resolved from there even though this file lives at the
 * repo root. The "Live engine" badge is matched with { exact: true } and
 * waited for, so the brief demo→live flash on cold start can't cause a false
 * pass. Assertions target the LIVE JSON shape.
 */
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { setTimeout as sleep } from "node:timers/promises";
import { connect } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const FRONTEND = join(ROOT, "frontend");
const require = createRequire(join(FRONTEND, "package.json"));
const { chromium } = require("playwright");

const API_PORT = Number(process.env.CREDITLENS_API_PORT || 8002);
const UI_PORT = 3003; // fixed by frontend/package.json `next dev -p 3003`
const PYTHON = process.env.CREDITLENS_PYTHON || "python3";
const API = `http://127.0.0.1:${API_PORT}`;
const UI = `http://127.0.0.1:${UI_PORT}`;

// The default applicant, mapped exactly as the frontend's toApiApplicant()
// sends it (frontend/lib/api.ts) — so the driver exercises the real wire shape.
const DEFAULT_APPLICANT = {
  annual_income: 75000, loan_amount: 12000, loan_term_months: 36,
  debt_to_income: 0.32, employment_years: 4, fico_score: 720,
  credit_utilization: 0.25, num_delinquencies: 0,
  months_since_delinquency: -1, num_hard_inquiries: 1, credit_age_months: 84,
  monthly_inflow: 6250, monthly_outflow: 4800, paycheck_regularity: 0.92,
  bill_payment_consistency: 0.88, nsf_count: 0, overdraft_days: 0,
  rent_payment_score: 0.95, utility_payment_score: 0.85,
  banking_tenure_months: 48, avg_account_balance: 8500,
  savings_rate: 0.1,
};

// Signature values (deterministic; seed=42 + the requirements.txt pins).
// Re-derived in-process 2026-11-04 (TestClient).
const EXPECT_SCORE = 993; // default applicant credit_score, "Approved"
const EXPECT_GROUP_A_N = 1234; // fairness test-set group A n_samples
const EXPECT_GROUP_SUM = 2000; // 20% test split of the 10,000-row synthetic set
const DISPARATE_IMPACT_FLOOR = 0.8; // documented 0.80 disparate-impact screen (README)
const VERDICTS = ["PASS", "FAIL", "INCONCLUSIVE"];
// The UI's label for each verdict (frontend/app/page.tsx VERDICT_COPY).
const VERDICT_LABEL = { PASS: "PASSES", FAIL: "FAILS", INCONCLUSIVE: "INCONCLUSIVE" };

// ---- pass/fail bookkeeping -------------------------------------------------
let pass = 0,
  fail = 0;
const ok = (n) => {
  pass++;
  console.log(`  ✓ ${n}`);
};
const bad = (n, e) => {
  fail++;
  console.log(`  ✗ ${n}${e ? " — " + e : ""}`);
};
async function check(name, fn) {
  try {
    await fn();
    ok(name);
  } catch (e) {
    bad(name, (e && e.message ? e.message : String(e)).split("\n")[0].slice(0, 180));
  }
}

// ---- server lifecycle -----------------------------------------------------
const children = [];
function launch(label, cmd, args, cwd, env = {}) {
  const child = spawn(cmd, args, {
    cwd,
    env: { ...process.env, ...env },
    detached: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.label = label;
  child.errtail = [];
  const keep = (d) => {
    for (const line of String(d).split("\n")) {
      if (line.trim()) {
        child.errtail.push(line);
        if (child.errtail.length > 30) child.errtail.shift();
      }
    }
  };
  child.stderr.on("data", keep);
  child.stdout.on("data", keep);
  child.on("error", (e) => keep(`spawn error: ${e.message}`));
  children.push(child);
  return child;
}
function killAll() {
  for (const c of children) {
    if (c.pid == null || c.exitCode !== null) continue;
    try {
      process.kill(-c.pid, "SIGKILL");
    } catch {
      try {
        c.kill("SIGKILL");
      } catch {}
    }
  }
}
for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, () => {
    killAll();
    process.exit(1);
  });
}

function portListening(port) {
  return new Promise((resolve) => {
    const s = connect({ host: "127.0.0.1", port }, () => {
      s.destroy();
      resolve(true);
    });
    s.on("error", () => resolve(false));
    s.setTimeout(1000, () => {
      s.destroy();
      resolve(false);
    });
  });
}
async function waitUntil(what, fn, timeoutMs, child) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (child && child.exitCode !== null) {
      throw new Error(`${child.label} exited early (code ${child.exitCode}): ${child.errtail.slice(-6).join(" | ")}`);
    }
    try {
      if (await fn()) return;
    } catch {}
    await sleep(400);
  }
  throw new Error(`timeout waiting for ${what} after ${timeoutMs}ms${child ? ": " + child.errtail.slice(-6).join(" | ") : ""}`);
}

let browser;
const t0 = Date.now();
try {
  // Preflight: refuse to run against a stale server on our ports (8002 and
  // 3003 by default): it could pass checks this code would fail.
  for (const [port, name] of [[API_PORT, "API"], [UI_PORT, "frontend"]]) {
    if (await portListening(port)) {
      throw new Error(`port ${port} (${name}) already in use — stop the stale process and re-run`);
    }
  }

  console.log("Starting servers:");
  console.log(`  API :${API_PORT} (${PYTHON}), UI :${UI_PORT}`);
  const apiProc = launch("api", PYTHON, ["-m", "uvicorn", "creditlens.api.app:app", "--host", "127.0.0.1", "--port", String(API_PORT)], ROOT);
  await check("API up (/health ok)", () =>
    waitUntil(
      "API /health",
      async () => {
        const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(2000) });
        return r.ok && (await r.json()).status === "ok";
      },
      30000,
      apiProc,
    ),
  );
  // Startup training populates the fairness report + model; wait for it so the
  // signature endpoints are live before we assert (not just the bare port).
  await check("startup training complete (/fairness/report ready)", () =>
    waitUntil(
      "fairness report",
      async () => (await fetch(`${API}/fairness/report`, { signal: AbortSignal.timeout(2000) })).ok,
      60000,
      apiProc,
    ),
  );

  // The UI must call THIS engine, whichever port it is on.
  const uiProc = launch("frontend", "npm", ["run", "dev"], FRONTEND, { NEXT_PUBLIC_API_URL: API });
  await check("frontend listening (:3003)", () => waitUntil("frontend port", () => portListening(UI_PORT), 60000, uiProc));

  // ---- API live-fire: deterministic score + honest fairness over HTTP ------
  console.log("API scoring + fairness:");
  const scoreRes = await fetch(`${API}/score`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(DEFAULT_APPLICANT),
    signal: AbortSignal.timeout(15000),
  });
  await check("POST /score -> 200", async () => {
    if (scoreRes.status !== 200) throw new Error(`HTTP ${scoreRes.status}`);
  });
  const score = await scoreRes.json();
  await check(`default applicant scores exactly ${EXPECT_SCORE} ("Approved")`, () => {
    if (score.credit_score !== EXPECT_SCORE) throw new Error(`credit_score ${score.credit_score} ≠ ${EXPECT_SCORE}`);
    if (score.decision !== "Approved") throw new Error(`decision ${score.decision} ≠ Approved`);
    // The probability must be the score's own (score = round((1 - p) * 1000));
    // p arrives rounded to 4 dp, so allow that rounding and nothing more.
    const implied = (1 - score.default_probability) * 1000;
    if (!(Math.abs(implied - score.credit_score) <= 0.55))
      throw new Error(`default_probability ${score.default_probability} implies ${implied.toFixed(1)}, not ${score.credit_score}`);
  });

  const frRes = await fetch(`${API}/fairness/report`, { signal: AbortSignal.timeout(15000) });
  const fr = await frRes.json();
  await check("fairness audit covers ALL 3 groups on the full test set (no cherry-picking)", () => {
    const groups = fr.group_metrics || {};
    for (const g of ["A", "B", "C"]) if (!groups[g]) throw new Error(`group ${g} missing from fairness report`);
    const sum = Object.values(groups).reduce((a, m) => a + m.n_samples, 0);
    if (sum !== EXPECT_GROUP_SUM) throw new Error(`group n_samples sum ${sum} ≠ ${EXPECT_GROUP_SUM}`);
    if (groups.A.n_samples !== EXPECT_GROUP_A_N) throw new Error(`group A n_samples ${groups.A.n_samples} ≠ ${EXPECT_GROUP_A_N}`);
  });
  await check(`disparate impact passes its ${DISPARATE_IMPACT_FLOOR} screen on its interval, not just its point`, () => {
    const di = fr.checks && fr.checks.disparate_impact;
    if (!di) throw new Error("checks.disparate_impact missing");
    if (!(di.ci[0] >= DISPARATE_IMPACT_FLOOR)) throw new Error(`DI interval lower bound ${di.ci[0]} < ${DISPARATE_IMPACT_FLOOR}`);
    if (di.verdict !== "PASS" || fr.passes_disparate_impact !== true) throw new Error(`DI verdict ${di.verdict}`);
  });
  await check("every criterion has an interval + 3-state verdict; the overall verdict follows from them", () => {
    const verdicts = [];
    for (const name of ["demographic_parity", "equal_opportunity", "equalized_odds", "disparate_impact"]) {
      const c = fr.checks[name];
      if (!c) throw new Error(`checks.${name} missing`);
      if (!VERDICTS.includes(c.verdict)) throw new Error(`${name} verdict ${c.verdict}`);
      if (!(c.ci[0] <= c.value && c.value <= c.ci[1])) throw new Error(`${name} ${c.value} outside its CI ${c.ci}`);
      verdicts.push(c.verdict);
    }
    const derived = verdicts.includes("FAIL") ? "FAIL" : verdicts.every((v) => v === "PASS") ? "PASS" : "INCONCLUSIVE";
    if (fr.overall_verdict !== derived) throw new Error(`overall_verdict ${fr.overall_verdict} ≠ derived ${derived}`);
  });

  const mRes = await fetch(`${API}/model/metrics`, { signal: AbortSignal.timeout(15000) });
  const metrics = await mRes.json();
  await check("model trained: auc_roc in a sane band [0.80, 0.95]", () => {
    if (!(metrics.auc_roc >= 0.8 && metrics.auc_roc <= 0.95)) throw new Error(`auc_roc ${metrics.auc_roc} out of band`);
  });

  // ---- UI live-fire: real browser, real backend, 0 console errors ----------
  console.log("UI (headless Chromium @ :3003):");
  browser = await chromium.launch();
  const page = await (await browser.newContext()).newPage();
  const consoleErrors = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));
  await page.goto(UI, { waitUntil: "domcontentloaded", timeout: 60000 });

  await check('dashboard shows "Live engine" (real backend, not demo)', async () => {
    // exact:true so this can't false-match other copy; waitFor blocks through
    // the brief demo→live flash on cold start until the badge settles live.
    await page.getByText("Live engine", { exact: true }).first().waitFor({ timeout: 30000 });
    const body = await page.innerText("body");
    if (body.includes("Demo data")) throw new Error("demo badge present");
  });

  // The live engine's deterministic EXPECT_SCORE. The no-backend page runs the same
  // recorded model and scores this applicant the same, so this alone does not prove
  // the engine answered: the "Live engine" badge above and the chart check below (live
  // Shapley values, where the no-backend page draws log-odds) do.
  // exact:true is load-bearing for non-vacuity — ScoreGauge renders the number
  // as its OWN text node (`text-5xl … {score}`), and a bare integer would else
  // substring-match unrelated page figures (e.g. avg_account_balance 8500 ⊃
  // "500"), so a tampered EXPECT_SCORE would falsely pass. With exact:true the
  // tamper makes THIS check fail too, not just the API one.
  await check(`UI renders the live score ${EXPECT_SCORE}`, async () => {
    await page.getByText(String(EXPECT_SCORE), { exact: true }).first().waitFor({ timeout: 15000 });
  });
  // The audit has no number until the engine answers: the gates render with it.
  await page.getByText("Live audit", { exact: true }).waitFor({ timeout: 60000 });
  const uiText = await page.innerText("body");
  await check("UI renders the model-wide fairness audit (all 4 gates)", () => {
    if (!uiText.includes("Fairness Audit")) throw new Error('"Fairness Audit" not rendered');
    for (const gate of ["Demographic Parity", "Equal Opportunity", "Equalized Odds", "Disparate Impact"]) {
      // overall_pass ANDs all four gates; every one it ANDs must be visible, or
      // a red FAILS badge sits over an incomplete explanation.
      if (!uiText.includes(gate)) throw new Error(`"${gate}" gate not rendered`);
    }
  });

  // The risk band and its observed default rate are the live audit's.
  await check("risk band and its observed default rate come from the live audit", async () => {
    const band = fr.score_bands.find((b) => EXPECT_SCORE >= b.min_score && EXPECT_SCORE <= b.max_score);
    if (!band) throw new Error("no band for the score");
    const label = (await page.getByTestId("risk-band").innerText()).trim();
    if (label !== band.label) throw new Error(`band "${label}" ≠ "${band.label}"`);
    const observed = (await page.getByTestId("band-observed").innerText()).trim();
    if (observed !== `${(band.observed_default_rate * 100).toFixed(1)}%`) throw new Error(`observed ${observed}`);
  });

  // The audit's calibration gap and the large draw are on the page, from the API.
  await check("fairness card shows each group's calibration gap and the large-draw note", async () => {
    await page.getByTestId("population-note").waitFor({ timeout: 15000 });
    const pop = fr.population;
    if (!pop || pop.n_samples !== 400000) throw new Error("population block missing");
    const note = await page.getByTestId("population-note").innerText();
    const c = pop.group_metrics.C;
    for (const want of ["400,000", `${(c.default_rate * 100).toFixed(1)}%`, `${(c.mean_predicted_default * 100).toFixed(1)}%`,
      `${(pop.checks.equalized_odds.value * 100).toFixed(1)}%`]) {
      if (!note.includes(want)) throw new Error(`note lacks ${want}: ${note}`);
    }
    const cell = await page.getByTestId("calibration-C").innerText();
    const gap = fr.group_metrics.C.calibration_in_the_large * 100;
    const shown = `${gap >= 0 ? "+" : "−"}${Math.abs(gap).toFixed(1)}`;
    if (!cell.includes(shown)) throw new Error(`calibration cell "${cell}" lacks ${shown}`);
  });

  // The lender's offered rate is not an input: the API rejects
  // it, and no "Interest Rate" factor can appear on the page.
  await check("interest rate is not an input: rejected by the API, absent from the chart", async () => {
    const res = await fetch(`${API}/score`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ...DEFAULT_APPLICANT, interest_rate: 0.12 }),
      signal: AbortSignal.timeout(15000),
    });
    if (res.status !== 422) throw new Error(`interest_rate accepted: HTTP ${res.status}`);
    const card = page.locator("div.bg-panel").filter({ hasText: "Feature Contributions" }).first();
    const labels = await card.locator("div.w-44").allInnerTexts();
    if (labels.some((l) => l.includes("Interest Rate"))) throw new Error("Interest Rate charted as a factor");
  });

  // The verdict badge must say exactly what the API reported — through all
  // three states. A two-way PASSES/FAILS switch would render INCONCLUSIVE as
  // FAILS, which is the claim this check exists to stop.
  await check(`fairness badge shows the API's verdict ("${VERDICT_LABEL[fr.overall_verdict]}") under "Live audit"`, async () => {
    const badge = page.getByTestId("fairness-verdict");
    await badge.waitFor({ timeout: 15000 });
    await page.getByText("Live audit", { exact: true }).waitFor({ timeout: 15000 });
    const text = (await badge.innerText()).trim();
    if (text !== VERDICT_LABEL[fr.overall_verdict]) throw new Error(`badge "${text}" ≠ "${VERDICT_LABEL[fr.overall_verdict]}"`);
  });

  // The contribution chart must be the LIVE attribution, drawn honestly: the
  // eight largest contributions by magnitude, in order, then one "All other
  // factors" row, so the bars sum to the applicant's whole move from the
  // baseline. The no-backend page's bars are log-odds, so they cannot pass this.
  await check("contribution chart = live attribution: top 8 by magnitude + remainder, summing to the move", async () => {
    const res = await fetch(`${API}/explain/0`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(DEFAULT_APPLICANT),
      signal: AbortSignal.timeout(15000),
    });
    const ex = await res.json();
    const ranked = Object.values(ex.shap_values).sort((a, b) => Math.abs(b) - Math.abs(a));
    const card = page.locator("div.bg-panel").filter({ hasText: "Feature Contributions" }).first();
    await card.getByTestId("chart-total").waitFor({ timeout: 15000 });
    const labels = (await card.locator("div.w-44").allInnerTexts()).map((t) => t.trim());
    const shown = (await card.locator("div.w-16.text-right").allInnerTexts()).map((t) =>
      parseFloat(t.trim().replace("−", "-")),
    );
    if (labels.length !== 9 || labels[8] !== "All other factors")
      throw new Error(`rows: ${labels.join(", ")}`);
    ranked.slice(0, 8).forEach((v, i) => {
      if (Math.abs(shown[i] - v) > 0.0006) throw new Error(`row ${i} ${shown[i]} != live ${v.toFixed(4)}`);
    });
    const move = ex.default_probability - ex.base_value; // served P minus baseline
    const sum = shown.reduce((a, b) => a + b, 0);
    if (Math.abs(sum - move) > 0.005) throw new Error(`bars sum ${sum.toFixed(4)} != move ${move.toFixed(4)}`);
  });

  await check("0 console errors", () => {
    if (consoleErrors.length) throw new Error(`${consoleErrors.length}: ${consoleErrors.slice(0, 3).join(" | ")}`);
  });

  // After the console check: a blocked request logs a console error by design. Under the
  // live badge the audit card shows no number until /fairness/report answers.
  await check("no audit number under the live badge while the report is pending or failed", async () => {
    const context = await browser.newContext();
    const second = await context.newPage();
    let release;
    const held = new Promise((resolve) => { release = resolve; });
    await second.route("**/fairness/report", async (route) => { await held; await route.abort(); });
    await second.goto(UI, { waitUntil: "domcontentloaded", timeout: 60000 });
    await second.getByText("Live engine", { exact: true }).first().waitFor({ timeout: 30000 });
    const refuse = async (when) => {
      await second.getByTestId("audit-pending").waitFor({ timeout: 15000 });
      await second.getByText("Audit pending", { exact: true }).waitFor({ timeout: 5000 });
      for (const absent of ["fairness-verdict", "population-note"]) {
        if (await second.getByTestId(absent).count()) throw new Error(`${absent} shown while ${when}`);
      }
      const body = await second.innerText("body");
      for (const absent of ["Demo audit", "Equalized Odds", "95% CI"]) {
        if (body.includes(absent)) throw new Error(`"${absent}" shown while ${when}`);
      }
    };
    await refuse("pending");
    release();
    await second.getByText("The engine did not return its audit", { exact: false }).waitFor({ timeout: 15000 });
    await refuse("failed");
    await second.unroute("**/fairness/report");
    await second.getByText("Live audit", { exact: true }).waitFor({ timeout: 30000 });
    await second.getByTestId("fairness-verdict").waitFor({ timeout: 5000 });
    if (await second.getByTestId("audit-pending").count()) throw new Error("pending note still shown");
    await context.close();
  });

  // One failed call must not strand the page, and while it is off the engine every label
  // must say so: the note and the audit's badge agree.
  await check("a lost /explain says demo everywhere, then recovers to live on its own", async () => {
    await page.route("**/explain/**", (route) => route.abort());
    await page.getByText("Median Applicant", { exact: true }).click();
    await page.getByText("Demo data", { exact: true }).first().waitFor({ timeout: 15000 });
    await page.getByTestId("demo-note").waitFor({ timeout: 5000 });
    await page.getByText("Demo audit", { exact: true }).waitFor({ timeout: 5000 });
    if (await page.getByText("Live audit", { exact: true }).count()) throw new Error('"Live audit" beside the demo note');
    await page.unroute("**/explain/**");
    await page.getByText("Live engine", { exact: true }).first().waitFor({ timeout: 30000 });
    await page.getByText("Live audit", { exact: true }).waitFor({ timeout: 30000 });
    await page.getByTestId("score-value").waitFor({ timeout: 30000 });
    if (await page.getByTestId("demo-note").count()) throw new Error("demo note still shown after recovery");
  });
} catch (e) {
  fail++;
  console.log(`\n✗ FATAL — ${e && e.message ? e.message : String(e)}`);
} finally {
  if (browser) {
    try {
      await browser.close();
    } catch {}
  }
  killAll();
}

console.log(`\n${pass} passed, ${fail} failed  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
process.exit(fail === 0 ? 0 : 1);
