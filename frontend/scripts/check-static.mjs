/**
 * Runs after `npm run build` (npm's postbuild hook): reads the prerendered page
 * and fails the build if it ships a score before any engine has answered, or if a
 * production build with no public URLs advertises localhost to crawlers. Next
 * resolves a relative og:image against localhost when metadataBase is missing, so
 * omitting the base alone is not enough.
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const html = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", ".next/server/app/index.html"), "utf8");
const problems = [];
// Before the engine answers there is no number: a locally computed score in the
// prerender would show under the "Live engine" badge until the first response.
if (html.includes('data-testid="score-value"')) problems.push("the prerendered page ships a score");
if (!html.includes('data-testid="score-pending"')) problems.push("the prerendered page lacks the pending state");
if (html.includes('data-testid="fairness-verdict"')) problems.push("the prerendered page ships a fairness verdict");
if (!(process.env.NEXT_PUBLIC_SITE_URL ?? "").trim()) {
  if (/localhost|127\.0\.0\.1/.test(html)) problems.push("the prerendered page names localhost");
  for (const tag of ["og:image", "twitter:image", "og:url"]) {
    if (html.includes(tag)) problems.push(`${tag} is set though NEXT_PUBLIC_SITE_URL is not`);
  }
}
if (problems.length) {
  console.error(`check-static FAILED: ${problems.join("; ")}`);
  process.exit(1);
}
console.log("check-static: the prerendered page advertises no localhost URL and ships no score");
