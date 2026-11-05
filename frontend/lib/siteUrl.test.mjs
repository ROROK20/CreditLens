// node --test lib/siteUrl.test.mjs  (npm test; Node 20+, no dependencies)
import assert from "node:assert/strict";
import test from "node:test";

import { resolveSiteUrl } from "./siteUrl.mjs";

test("a production build without NEXT_PUBLIC_SITE_URL advertises no localhost URL", () => {
  assert.equal(resolveSiteUrl(undefined, "production"), null);
  assert.equal(resolveSiteUrl(" ", "production"), null);
});

test("next dev falls back to the local server; a configured origin is used", () => {
  assert.equal(resolveSiteUrl(undefined, "development"), "http://localhost:3003");
  assert.equal(resolveSiteUrl("https://creditlens.example/", "production"), "https://creditlens.example");
});
