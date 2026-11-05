/**
 * The engine's served model, run from its recorded weights (engineSnapshot.json),
 * for when no engine can be reached. The launch deploy has no backend, so this IS
 * what a visitor sees. The served model is a logistic regression, so a browser can
 * evaluate it exactly: for the applicants recorded in the snapshot this returns the
 * engine's own score, it refuses what the engine refuses, and it warns where the
 * engine warns.
 *
 * `npm test` checks this against the snapshot's recorded answers, and
 * tests/test_engine_snapshot.py checks the snapshot against the live engine.
 */
import snapshot from "./engineSnapshot.json" with { type: "json" };

const RECENCY_PLATEAU_MONTHS = 24; // creditlens/features/engineering.py

/** The model's features, derived as creditlens/features/engineering.py derives them. */
export function engineFeatures(a) {
  const months = a.months_since_delinquency;
  return {
    fico_score: a.fico_score,
    debt_to_income: a.debt_to_income,
    credit_utilization: a.credit_utilization,
    num_delinquencies: a.num_delinquencies,
    months_since_delinquency: months >= 0 ? Math.min(months, RECENCY_PLATEAU_MONTHS) : RECENCY_PLATEAU_MONTHS,
    num_hard_inquiries: a.num_hard_inquiries,
    credit_age_months: a.credit_age_months,
    nsf_count: a.nsf_count,
    overdraft_days: a.overdraft_days,
    bill_payment_consistency: a.bill_payment_consistency,
    paycheck_regularity: a.paycheck_regularity,
    rent_payment_score: a.rent_payment_score,
    utility_payment_score: a.utility_payment_score,
    monthly_net_cashflow: a.monthly_inflow - a.monthly_outflow,
    savings_rate: a.savings_rate,
    avg_account_balance: Math.log1p(a.avg_account_balance / 1000),
    banking_tenure_months: a.banking_tenure_months,
    employment_years: a.employment_years,
    monthly_payment_burden: (a.loan_amount / a.loan_term_months) / (a.annual_income / 12),
  };
}

/** P(default) and the score, by the engine's weights and its score rule. */
export function engineScore(a) {
  const x = engineFeatures(a);
  let logOdds = snapshot.model.intercept;
  for (const [name, weight] of Object.entries(snapshot.model.weights)) logOdds += weight * x[name];
  const p = 1 / (1 + Math.exp(-logOdds));
  return { defaultProbability: p, creditScore: Math.round((1 - p) * 1000) };
}

/** What the engine's field bounds and joint rules refuse (a 422 live); empty when it
 *  would score. */
export function engineRefusals(a) {
  const r = snapshot.rules;
  const monthly = a.annual_income / 12;
  const out = [];
  for (const [field, [lo, hi]] of Object.entries(r.fields)) {
    const v = a[field];
    if (field === "months_since_delinquency" && v === -1) continue; // "never delinquent"
    if (v < lo || v > hi) out.push(`${field} must be ${lo} to ${hi}; it is ${v}`);
  }
  for (const field of r.integer_fields) {
    if (!Number.isInteger(a[field])) out.push(`${field} must be a whole number; it is ${a[field]}`);
  }
  if ((a.num_delinquencies === 0) !== (a.months_since_delinquency === -1)) {
    out.push("months_since_delinquency must be -1 exactly when num_delinquencies is 0");
  }
  const inflow = a.monthly_inflow / monthly;
  if (inflow < r.inflow_to_monthly_income[0] || inflow > r.inflow_to_monthly_income[1]) {
    out.push(`monthly_inflow must be ${r.inflow_to_monthly_income.join("-")} x annual_income / 12; it is ${inflow.toFixed(2)} x`);
  }
  const outflow = a.monthly_outflow / a.monthly_inflow;
  if (outflow < r.outflow_to_inflow[0] || outflow > r.outflow_to_inflow[1]) {
    out.push(`monthly_outflow must be ${r.outflow_to_inflow.join("-")} x monthly_inflow; it is ${outflow.toFixed(2)} x`);
  }
  const balance = a.avg_account_balance / monthly;
  if (balance < r.balance_to_monthly_income[0] || balance > r.balance_to_monthly_income[1]) {
    out.push(`avg_account_balance must be ${r.balance_to_monthly_income.join("-")} x annual_income / 12; it is ${balance.toFixed(2)} x`);
  }
  const net = a.monthly_inflow - a.monthly_outflow;
  if (net < r.net_cashflow[0] || net > r.net_cashflow[1]) {
    out.push(`monthly_inflow - monthly_outflow must be ${r.net_cashflow.join(" to ")}; it is ${net.toFixed(2)}`);
  }
  const burden = (a.loan_amount / a.loan_term_months) / monthly;
  if (burden > r.max_payment_burden) {
    out.push(`the monthly loan payment must be at most ${r.max_payment_burden} x monthly income; it is ${burden.toFixed(2)} x`);
  }
  return out;
}

// Python's "%g", for every value the page can produce (under 1e6, six figures).
const g = (v) => String(Number(v.toPrecision(6)));

/** What the engine's `warnings` says: creditlens/data/domain.py domain_warnings. */
export function engineWarnings(a) {
  const r = snapshot.rules;
  const values = { ...a, monthly_net_cashflow: a.monthly_inflow - a.monthly_outflow };
  const out = [];
  for (const [field, [lo, hi]] of Object.entries(r.warning_band)) {
    const v = values[field];
    if (field === "months_since_delinquency" && v < 0) continue;
    if (v < lo || v > hi) {
      out.push(`${field} ${g(v)} is ${v < lo ? `below ${g(lo)}` : `above ${g(hi)}`}, outside the central 99% of ` +
        "the training data: few training applicants look like this, so treat the score as less certain");
    }
  }
  const monthly = a.annual_income / 12;
  const inflow = a.monthly_inflow / monthly;
  const outflow = a.monthly_outflow / a.monthly_inflow;
  const balance = a.avg_account_balance / monthly;
  for (const [name, ratio, [lo, hi]] of [
    ["monthly_inflow / (annual_income / 12)", inflow, r.inflow_warning],
    ["monthly_outflow / monthly_inflow", outflow, r.outflow_warning],
    ["avg_account_balance / (annual_income / 12)", balance, r.balance_warning],
  ]) {
    if (ratio < lo || ratio > hi) {
      out.push(`${name} is ${ratio.toFixed(3)}, ${ratio < lo ? `below ${g(lo)}` : `above ${g(hi)}`}, outside the ` +
        "central 99% of the training data: treat the score as less certain");
    }
  }
  const burden = (a.loan_amount / a.loan_term_months) / monthly;
  if (burden > r.payment_burden_warning) {
    out.push(`the monthly loan payment is ${burden.toFixed(2)}x monthly income, above ${g(r.payment_burden_warning)} ` +
      "(the central 99% of the training data): treat the score as less certain");
  }
  const edge = r.cash_flow_edge;
  const atEdge = [
    ["inflow", inflow >= edge.inflow_to_monthly_income],
    ["outflow", outflow <= edge.outflow_to_inflow],
    ["balance", balance >= edge.balance_to_monthly_income],
  ].filter(([, on]) => on).map(([name]) => name);
  if (atEdge.length >= 2) {
    out.push(`${atEdge.join(", ")} are each in the most favourable tenth of the training data at once: these ` +
      "figures are self-reported and stated cash flow can be written to pass, so verify them independently " +
      "before relying on this score");
  }
  return out;
}

// Each factor the web chart names -> the model features it stands for. The payment
// burden spans loan amount, term and income, and is drawn under the loan amount.
const FACTORS = {
  fico_score: ["fico_score"], credit_utilization: ["credit_utilization"], debt_to_income: ["debt_to_income"],
  delinquency_history: ["num_delinquencies", "months_since_delinquency"], num_hard_inquiries: ["num_hard_inquiries"],
  credit_age_months: ["credit_age_months"], loan_amount: ["monthly_payment_burden"],
  paycheck_regularity: ["paycheck_regularity"], bill_payment_consistency: ["bill_payment_consistency"],
  nsf_count: ["nsf_count"], overdraft_days: ["overdraft_days"], employment_years: ["employment_years"],
  savings_rate: ["savings_rate"], income_cash_flow_and_balance: ["monthly_net_cashflow", "avg_account_balance"],
  rent_payment_score: ["rent_payment_score"], utility_payment_score: ["utility_payment_score"],
  banking_tenure_months: ["banking_tenure_months"],
};

/** Each factor's change in log-odds against the average training applicant. Exact
 *  for this linear model, and they sum to its log-odds minus the average's. */
export function engineFactors(a) {
  const x = engineFeatures(a);
  const { weights, feature_means: means } = snapshot.model;
  return Object.entries(FACTORS).map(([factor, features]) => ({
    feature: factor,
    contribution: features.reduce((sum, f) => sum + weights[f] * (x[f] - means[f]), 0),
  }));
}

export { snapshot };
