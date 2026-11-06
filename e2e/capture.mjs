/**
 * Regenerate BOTH published images, each from the state it is documented to show:
 *   docs/screenshot.png      the README screenshot (1440 px wide, full page, 2x), from
 *                            a production build against the live engine
 *   frontend/public/og.png   the social-preview image (1200 x 630 at 2x), from a build
 *                            with no backend, which is what a frontend-only deploy
 *                            shows the visitor who follows the link
 * and docs/capture.json.
 *
 * What each page RENDERED is read from the DOM around the moment of its screenshot and
 * must equal the engine's answer (live) or its recording (no backend). That record goes
 * inside the PNG itself as a tEXt chunk, with an id shared by both images of one run,
 * and the file's SHA-256 goes in capture.json: tests/test_capture.py reads the record
 * out of the image, so editing the JSON cannot make a stale image pass, and an image
 * from another run does not pair. There is no OCR: the pixels are bound to the record
 * by the hash and by being taken between two reads of the page.
 *
 *   node e2e/capture.mjs      # or: make capture   (from the repo root)
 *   CREDITLENS_API_PORT=8012 CREDITLENS_PYTHON=python3.12 node e2e/capture.mjs
 *
 * It builds the frontend twice, starts the API and `next start`, verifies, captures,
 * and tears everything down. Not while `next dev` runs in this frontend.
 */
import { execFileSync, spawn } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { createRequire } from "node:module";
import { readFileSync, writeFileSync } from "node:fs";
import { connect } from "node:net";
import { dirname, join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const FRONTEND = join(ROOT, "frontend");
const require = createRequire(join(FRONTEND, "package.json"));
const { chromium } = require("playwright");

const API_PORT = Number(process.env.CREDITLENS_API_PORT || 8002);
const PYTHON = process.env.CREDITLENS_PYTHON || "python3";
const API = `http://127.0.0.1:${API_PORT}`;
const UI = "http://127.0.0.1:3003";
const SHOT = "docs/screenshot.png";
const OG = "frontend/public/og.png";
const RECORD_KEY = "creditlens-capture"; // tests/test_capture.py reads this tEXt chunk

// frontend/lib/scoring.ts DEFAULT_APPLICANT, as frontend/lib/api.ts sends it.
const DEFAULT_APPLICANT = {
  annual_income: 75000, loan_amount: 12000, loan_term_months: 36,
  debt_to_income: 0.32, employment_years: 4, fico_score: 720,
  credit_utilization: 0.25, num_delinquencies: 0,
  months_since_delinquency: -1, num_hard_inquiries: 1, credit_age_months: 84,
  monthly_inflow: 6250, monthly_outflow: 4800, paycheck_regularity: 0.92,
  bill_payment_consistency: 0.88, nsf_count: 0, overdraft_days: 0,
  rent_payment_score: 0.95, utility_payment_score: 0.85,
  banking_tenure_months: 48, avg_account_balance: 8500, savings_rate: 0.1,
};
const VERDICT_FROM_LABEL = { PASSES: "PASS", FAILS: "FAIL", INCONCLUSIVE: "INCONCLUSIVE" };

const children = new Set();
function launch(cmd, args, cwd, env = process.env) {
  const child = spawn(cmd, args, { cwd, env, detached: true, stdio: "ignore" });
  children.add(child);
  return child;
}
function stop(child) {
  children.delete(child);
  try { process.kill(-child.pid, "SIGKILL"); } catch { try { child.kill("SIGKILL"); } catch {} }
}
const killAll = () => { for (const c of [...children]) stop(c); };
for (const sig of ["SIGINT", "SIGTERM"]) process.on(sig, () => { killAll(); process.exit(1); });

const listening = (port) => new Promise((resolve) => {
  const s = connect({ host: "127.0.0.1", port }, () => { s.destroy(); resolve(true); });
  s.on("error", () => resolve(false));
});
async function up(url, ms) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try { if ((await fetch(url, { signal: AbortSignal.timeout(2000) })).ok) return; } catch {}
    await sleep(500);
  }
  throw new Error(`${url} not up after ${ms} ms`);
}
async function portFree(port, ms) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    if (!(await listening(port))) return;
    await sleep(250);
  }
  throw new Error(`port ${port} still busy after ${ms} ms`);
}
const sha256 = (path) => createHash("sha256").update(readFileSync(join(ROOT, path))).digest("hex");
const pct = (p) => `${(p * 100).toFixed(1)}%`;

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(buffer) {
  let c = 0xffffffff;
  for (const byte of buffer) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
/** Write `record` into the PNG as a tEXt chunk, right after IHDR. */
function embedRecord(path, record) {
  const png = readFileSync(join(ROOT, path));
  const body = Buffer.concat([Buffer.from("tEXt"), Buffer.from(`${RECORD_KEY}\0${JSON.stringify(record)}`, "latin1")]);
  const length = Buffer.alloc(4);
  length.writeUInt32BE(body.length - 4);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  const afterIhdr = 8 + 25; // signature, then IHDR's length + type + 13 bytes + CRC
  writeFileSync(join(ROOT, path), Buffer.concat([png.subarray(0, afterIhdr), length, body, crc, png.subarray(afterIhdr)]));
}

async function rendered(page) {
  return {
    credit_score: Number((await page.getByTestId("score-value").innerText()).trim()),
    decision: (await page.getByTestId("score-decision").innerText()).trim().toUpperCase(),
    default_probability: (await page.getByTestId("default-probability").innerText()).trim(),
    risk_band: (await page.getByTestId("risk-band").innerText()).trim(),
    fairness_verdict: VERDICT_FROM_LABEL[(await page.getByTestId("fairness-verdict").innerText()).trim()],
  };
}

/** Open the page, require it to show `expected`, screenshot it, require it still does. */
async function take(browser, { viewport, badge, audit, expected, path, fullPage }) {
  const context = await browser.newContext({ viewport, deviceScaleFactor: 2 });
  const page = await context.newPage();
  const errors = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(UI, { waitUntil: "networkidle", timeout: 90000 });
  await page.getByText(badge, { exact: true }).first().waitFor({ timeout: 60000 });
  await page.getByText(audit, { exact: true }).waitFor({ timeout: 60000 });
  await page.getByTestId("score-value").getByText(String(expected.credit_score), { exact: true }).waitFor({ timeout: 60000 });
  await page.getByTestId("fairness-verdict").waitFor({ timeout: 60000 });
  await page.waitForTimeout(1500); // the score rail settles
  const before = await rendered(page);
  await page.screenshot({ path: join(ROOT, path), fullPage });
  const after = await rendered(page);
  for (const [key, want] of Object.entries(expected)) {
    if (before[key] !== want || after[key] !== want) throw new Error(`${path}: ${key} rendered ${before[key]} / ${after[key]}, expected ${want}`);
  }
  if (errors.length) throw new Error(`${path}: console errors: ${errors.join(" | ")}`);
  await context.close();
  return after;
}

function expectedFrom(answer, verdict, bands) {
  const band = bands.find((b) => answer.credit_score >= b.min_score && answer.credit_score <= b.max_score);
  return {
    credit_score: answer.credit_score,
    decision: answer.decision.toUpperCase(),
    default_probability: pct(answer.default_probability),
    risk_band: band.label,
    fairness_verdict: verdict,
  };
}

function buildEnv(apiUrl) {
  const env = { ...process.env };
  delete env.NEXT_PUBLIC_API_URL;
  delete env.NEXT_PUBLIC_SITE_URL;
  if (apiUrl) env.NEXT_PUBLIC_API_URL = apiUrl;
  return env;
}

let browser;
let failed = false;
try {
  for (const port of [API_PORT, 3003]) {
    if (await listening(port)) throw new Error(`port ${port} already in use: stop that process, or set CREDITLENS_API_PORT`);
  }
  browser = await chromium.launch();
  const captureId = randomBytes(8).toString("hex");
  const captured = execFileSync("date", ["+%Y-%m-%d"]).toString().trim();

  // The README screenshot: a production build against the live engine.
  console.log(`building the frontend with NEXT_PUBLIC_API_URL=${API}`);
  execFileSync("npm", ["run", "build"], { cwd: FRONTEND, stdio: "inherit", env: buildEnv(API) });
  const api = launch(PYTHON, ["-m", "uvicorn", "creditlens.api.app:app", "--host", "127.0.0.1", "--port", String(API_PORT)], ROOT);
  let web = launch("npx", ["next", "start", "-p", "3003", "-H", "127.0.0.1"], FRONTEND, buildEnv(API));
  await up(`${API}/fairness/report`, 120000);
  await up(UI, 120000);
  const score = await (await fetch(`${API}/score`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(DEFAULT_APPLICANT),
  })).json();
  const report = await (await fetch(`${API}/fairness/report`)).json();
  const liveShows = await take(browser, {
    viewport: { width: 1440, height: 900 }, badge: "Live engine", audit: "Live audit", path: SHOT, fullPage: true,
    expected: expectedFrom(score, report.overall_verdict, report.score_bands),
  });
  stop(web);
  stop(api);
  await portFree(3003, 15000);

  // The social image: what a deploy with no backend shows.
  console.log("building the frontend with no public URLs");
  execFileSync("npm", ["run", "build"], { cwd: FRONTEND, stdio: "inherit", env: buildEnv(null) });
  web = launch("npx", ["next", "start", "-p", "3003", "-H", "127.0.0.1"], FRONTEND, buildEnv(null));
  await up(UI, 120000);
  const snapshot = JSON.parse(readFileSync(join(FRONTEND, "lib/engineSnapshot.json"), "utf8"));
  const recorded = snapshot.vectors.find((v) => v.name === "default");
  const ogShows = await take(browser, {
    viewport: { width: 1200, height: 630 }, badge: "Demo data", audit: "Demo audit", path: OG, fullPage: false,
    expected: expectedFrom(recorded, snapshot.fairness.verdict, snapshot.score_bands),
  });
  stop(web);

  const images = [
    ["screenshot", SHOT, "live engine", liveShows],
    ["og", OG, "no backend", ogShows],
  ];
  const manifest = { captured, capture_id: captureId };
  for (const [key, path, state, shows] of images) {
    embedRecord(path, { capture_id: captureId, captured, state, shows });
    manifest[key] = { path, sha256: sha256(path), state, shows };
  }
  writeFileSync(join(ROOT, "docs/capture.json"), JSON.stringify(manifest, null, 2) + "\n");
  console.log(JSON.stringify(manifest, null, 2));
} catch (e) {
  failed = true;
  console.error(`capture FAILED: ${e && e.message ? e.message : e}`);
} finally {
  if (browser) { try { await browser.close(); } catch {} }
  killAll();
}
process.exit(failed ? 1 : 0);
