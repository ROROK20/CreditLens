/**
 * A production build with no public URLs, in a real browser: it must call no
 * engine, say it is showing demo data, and advertise no localhost URL.
 *
 *   node e2e/offline.mjs      # from the repo root; exit 0 = all pass
 *
 * It builds the frontend with NEXT_PUBLIC_API_URL and NEXT_PUBLIC_SITE_URL unset,
 * runs `next start` on :3003, records every request the page makes, and tears the
 * server down. Not while `next dev` runs in this frontend.
 */
import { execFileSync, spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const FRONTEND = join(ROOT, "frontend");
const require = createRequire(join(FRONTEND, "package.json"));
const { chromium } = require("playwright");
const UI = "http://127.0.0.1:3003";

const env = { ...process.env };
delete env.NEXT_PUBLIC_API_URL;
delete env.NEXT_PUBLIC_SITE_URL;

let server, browser;
const problems = [];
try {
  execFileSync("npm", ["run", "build"], { cwd: FRONTEND, stdio: "inherit", env });
  server = spawn("npx", ["next", "start", "-p", "3003", "-H", "127.0.0.1"], { cwd: FRONTEND, env, detached: true, stdio: "ignore" });
  for (let i = 0; i < 120; i++) {
    try { if ((await fetch(UI, { signal: AbortSignal.timeout(1000) })).ok) break; } catch {}
    await sleep(500);
  }
  const html = await (await fetch(UI)).text();
  if (/localhost|127\.0\.0\.1/.test(html)) problems.push("the served HTML names localhost");
  for (const tag of ["og:image", "twitter:image", "og:url"]) if (html.includes(tag)) problems.push(`${tag} present`);

  browser = await chromium.launch();
  const page = await (await browser.newContext()).newPage();
  const foreign = [];
  page.on("request", (r) => { if (!r.url().startsWith(UI) && !r.url().startsWith("data:")) foreign.push(r.url()); });
  await page.goto(UI, { waitUntil: "networkidle", timeout: 60000 });
  await page.getByText("Demo data", { exact: true }).first().waitFor({ timeout: 15000 });
  await page.getByTestId("demo-note").waitFor({ timeout: 15000 });
  await page.waitForTimeout(3000); // anything a health probe would still send
  if (foreign.length) problems.push(`requests left the site: ${foreign.join(", ")}`);
  const body = await page.innerText("body");
  if (!body.includes("Demo audit")) problems.push('"Demo audit" not shown');
  if (body.includes("Live engine")) problems.push('"Live engine" shown');
  // What a visitor to the no-backend deploy sees is the engine's own answer.
  const snapshot = JSON.parse(readFileSync(join(FRONTEND, "lib/engineSnapshot.json"), "utf8"));
  const recorded = snapshot.vectors.find((v) => v.name === "default");
  const shown = (await page.getByTestId("score-value").innerText()).trim();
  if (shown !== String(recorded.credit_score)) problems.push(`score ${shown} is not the engine's ${recorded.credit_score}`);
  await page.getByTestId("population-note").waitFor({ timeout: 15000 });
  await page.getByText("Weak Profile", { exact: true }).click();
  const weak = snapshot.vectors.find((v) => v.name === "weak");
  await page.getByTestId("score-value").getByText(String(weak.credit_score), { exact: true }).waitFor({ timeout: 15000 });
  // The engine's warnings show here too.
  const shownWarnings = await page.getByTestId("domain-warnings").locator("li").allInnerTexts();
  if (JSON.stringify(shownWarnings) !== JSON.stringify(weak.warnings)) problems.push(`warnings ${JSON.stringify(shownWarnings)}`);
} catch (e) {
  problems.push(e && e.message ? e.message : String(e));
} finally {
  if (browser) { try { await browser.close(); } catch {} }
  if (server) { try { process.kill(-server.pid, "SIGKILL"); } catch {} }
}
if (problems.length) {
  console.error(`offline FAILED: ${problems.join("; ")}`);
  process.exit(1);
}
console.log('offline: no request left the site; "Demo data" and the note show; the score is the engine\'s recorded answer; no localhost URL advertised');
