// node --test lib/engineModel.test.mjs  (npm test; Node 20+, no dependencies)
import assert from "node:assert/strict";
import test from "node:test";

import { engineFactors, engineRefusals, engineScore, engineWarnings, snapshot } from "./engineModel.mjs";

const scored = snapshot.vectors.filter((v) => v.status === 200);
const refused = snapshot.vectors.filter((v) => v.status === 422);

test("the browser model returns the engine's recorded answer for every recorded applicant", () => {
  assert.ok(scored.length >= 8);
  for (const v of scored) {
    const { creditScore, defaultProbability } = engineScore(v.applicant);
    assert.equal(creditScore, v.credit_score, v.name);
    assert.ok(Math.abs(defaultProbability - v.default_probability) < 5.1e-5, v.name);
    assert.deepEqual(engineRefusals(v.applicant), [], v.name);
  }
});

test("the browser model refuses what the engine refuses", () => {
  assert.ok(refused.length >= 5);
  for (const v of refused) assert.ok(engineRefusals(v.applicant).length > 0, v.name);
});

test("the browser model refuses every field outside its bound, not only the joint rules", () => {
  const base = scored.find((v) => v.name === "default").applicant;
  for (const [field, [lo, hi]] of Object.entries(snapshot.rules.fields)) {
    if (field === "months_since_delinquency") {
      assert.ok(engineRefusals({ ...base, num_delinquencies: 1, months_since_delinquency: hi + 1 }).length > 0, field);
      continue;
    }
    assert.ok(engineRefusals({ ...base, [field]: hi + 1 }).length > 0, `${field} above`);
    assert.ok(engineRefusals({ ...base, [field]: lo - 1 }).length > 0, `${field} below`);
  }
  assert.deepEqual(engineRefusals({ ...base, annual_income: 15000, monthly_inflow: 1250, monthly_outflow: 960,
    avg_account_balance: 465 }), ["avg_account_balance must be 480 to 170000; it is 465"]);
});

test("the browser model warns exactly where the engine warned", () => {
  assert.ok(scored.filter((v) => v.warnings.length > 0).length >= 5);
  for (const v of scored) assert.deepEqual(engineWarnings(v.applicant), v.warnings, v.name);
});

test("the factors sum to the applicant's log-odds minus the average applicant's", () => {
  const { weights, intercept, feature_means: means } = snapshot.model;
  const average = Object.entries(weights).reduce((sum, [f, w]) => sum + w * means[f], intercept);
  for (const v of scored) {
    const { defaultProbability: p } = engineScore(v.applicant);
    const sum = engineFactors(v.applicant).reduce((total, f) => total + f.contribution, 0);
    assert.ok(Math.abs(sum - (Math.log(p / (1 - p)) - average)) < 1e-9, v.name);
  }
});

test("risk bands order by the default rate observed in them", () => {
  const rates = snapshot.score_bands.map((b) => b.observed_default_rate);
  for (let i = 1; i < rates.length; i++) assert.ok(rates[i] > rates[i - 1], String(rates));
});
