// node --test lib/apiUrl.test.mjs  (npm test; Node 20+, no dependencies)
import assert from "node:assert/strict";
import test from "node:test";

import { resolveApiUrl } from "./apiUrl.mjs";

test("a production build without NEXT_PUBLIC_API_URL calls no engine", () => {
  // Not the VISITOR's localhost:8002.
  assert.equal(resolveApiUrl(undefined, "production"), null);
  assert.equal(resolveApiUrl("", "production"), null); // Dockerfile: ARG with no value
  assert.equal(resolveApiUrl("   ", "test"), null);
});

test("next dev without the variable uses the local engine", () => {
  assert.equal(resolveApiUrl(undefined, "development"), "http://localhost:8002");
});

test("a configured URL is used as given, trailing slash dropped", () => {
  assert.equal(resolveApiUrl("https://api.example.com/", "production"), "https://api.example.com");
  assert.equal(resolveApiUrl("http://127.0.0.1:8012", "development"), "http://127.0.0.1:8012");
});
