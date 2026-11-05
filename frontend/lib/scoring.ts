/**
 * Shared types and helpers for the page, and the no-backend fallback: when no engine
 * can be reached the page runs the engine's own served model from its recorded weights
 * (lib/engineModel.mjs). Scores match the engine's; the factors are log-odds against the
 * average applicant, not the live Shapley values, and the UI says so whenever it shows.
 */
import { engineFactors, engineScore, snapshot } from "./engineModel.mjs";
import { toApiApplicant } from "./api";

// Everything the engine's model uses, supplied by the user through the form
// (the API rejects an applicant with any of these missing). Nothing is filled
// in for the user. There is no interest-rate input: the offered rate is the
// lender's own price, an outcome of the decision, so the model does not use it.
export interface ApplicantInput {
  annualIncome: number;
  loanAmount: number;
  loanTermMonths: number;
  ficoScore: number;
  creditUtilization: number;
  debtToIncome: number;
  employmentYears: number;
  numDelinquencies: number;
  // Months since the most recent delinquency. Sent to the engine only when
  // numDelinquencies > 0; otherwise the applicant is "never delinquent" (-1),
  // the only combination the model was trained on (the API rejects others).
  monthsSinceDelinquency: number;
  numHardInquiries: number;
  creditAgeMonths: number;
  monthlyInflow: number;
  monthlyOutflow: number;
  paycheckRegularity: number;
  billPaymentConsistency: number;
  nsfCount: number;
  overdraftDays: number;
  rentPaymentScore: number;
  utilityPaymentScore: number;
  avgAccountBalance: number;
  bankingTenureMonths: number;
  savingsRate: number;
}

export interface FeatureContribution {
  feature: string;
  label: string;
  value: number;
  contribution: number; // positive = increases default risk, negative = decreases
  category: "core" | "behavioral" | "alternative" | "other";
}

/** Rows the contribution chart draws. */
export const CHART_ROWS = 8;
export const OTHER_LABEL = "All other factors";

/**
 * The chart's rows: the CHART_ROWS largest contributions by magnitude, then one
 * "All other factors" row holding the sum of the rest — so the bars add up to
 * the applicant's whole move from the baseline.
 */
export function chartRows(all: FeatureContribution[], n = CHART_ROWS): FeatureContribution[] {
  const sorted = [...all].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const shown = sorted.slice(0, n);
  const rest = sorted.slice(n).reduce((sum, c) => sum + c.contribution, 0);
  if (sorted.length > n && Math.abs(rest) > 0) {
    shown.push({ feature: "_other", label: OTHER_LABEL, value: 0, contribution: rest, category: "other" });
  }
  return shown;
}

export interface ScoringResult {
  creditScore: number; // 0-1000
  defaultProbability: number; // 0-1
  decision: "Approved" | "Review" | "Denied";
  decisionReason: string;
  contributions: FeatureContribution[];
  topPositives: FeatureContribution[];
  topNegatives: FeatureContribution[];
}

/**
 * Decision cutoffs — the SINGLE SOURCE OF TRUTH for the frontend, aligned to the
 * backend (creditlens/api/app.py `_get_decision` / `/score/batch`):
 *   score >= 600            → Approved
 *   450 <= score < 600      → Review
 *   score < 450             → Denied
 */
export const APPROVE_CUTOFF = 600;
export const REVIEW_CUTOFF = 450;

/** Mirrors creditlens/decision.py DECISION_REASONS (tests/test_decision.py). */
export const DECISION_REASONS: Record<ScoringResult["decision"], string> = {
  Approved: "Score at or above the approval cutoff (600).",
  Review: "Score in the manual-review band (450-599).",
  Denied: "Score below the review cutoff (450).",
};

export function scoreToDecision(score: number): ScoringResult["decision"] {
  if (score >= APPROVE_CUTOFF) return "Approved";
  if (score >= REVIEW_CUTOFF) return "Review";
  return "Denied";
}

// A risk band and what the audit set observed in it (creditlens/decision.py
// RISK_BANDS; tests/test_engine_snapshot.py requires the rates to order).
export interface ScoreBand {
  label: string;
  min_score: number;
  max_score: number;
  n: number;
  share: number;
  observed_default_rate: number | null;
}

export function bandForScore(score: number, bands: ScoreBand[]): ScoreBand | null {
  return bands.find((b) => score >= b.min_score && score <= b.max_score) ?? null;
}

// The engine's income-inflow rule (creditlens/data/domain.py): monthly inflow
// must be this multiple of annual income / 12. tests/test_docs.py pins it.
export const INFLOW_TO_MONTHLY_INCOME = [0.78, 1.22] as const;
// Its other two: outflow as a share of inflow, balance as a multiple of income / 12.
export const OUTFLOW_TO_INFLOW = [0.47, 0.98] as const;
export const BALANCE_TO_MONTHLY_INCOME = [0.37, 3.13] as const;

export const DEFAULT_APPLICANT: ApplicantInput = {
  annualIncome: 75_000,
  loanAmount: 12_000,
  loanTermMonths: 36,
  ficoScore: 720,
  creditUtilization: 0.25,
  debtToIncome: 0.32,
  employmentYears: 4,
  numDelinquencies: 0,
  monthsSinceDelinquency: 24,
  numHardInquiries: 1,
  creditAgeMonths: 84,
  monthlyInflow: 6_250,
  monthlyOutflow: 4_800,
  paycheckRegularity: 0.92,
  billPaymentConsistency: 0.88,
  nsfCount: 0,
  overdraftDays: 0,
  rentPaymentScore: 0.95,
  utilityPaymentScore: 0.85,
  avgAccountBalance: 8_500,
  bankingTenureMonths: 48,
  savingsRate: 0.1,
};

/** The no-backend answer: the engine's recorded model, run in the browser. */
export function scoreApplicant(applicant: ApplicantInput): ScoringResult {
  const wire = toApiApplicant(applicant);
  const { creditScore, defaultProbability } = engineScore(wire);
  const decision = scoreToDecision(creditScore);
  const contributions: FeatureContribution[] = engineFactors(wire).map((f: { feature: string; contribution: number }) => ({
    feature: f.feature,
    label: apiFeatureLabel(f.feature),
    value: 0,
    contribution: f.contribution,
    category: apiFeatureCategory(f.feature),
  }));
  const sorted = [...contributions].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return {
    creditScore,
    defaultProbability,
    decision,
    decisionReason: DECISION_REASONS[decision],
    contributions: chartRows(contributions),
    topPositives: sorted.filter((c) => c.contribution < 0).slice(0, 5),
    topNegatives: sorted.filter((c) => c.contribution > 0).slice(0, 5),
  };
}

// Fairness audit (creditlens/fairness/metrics.py). Three verdicts, not two:
// FAIL only when a criterion's confidence interval lies beyond its threshold,
// PASS only when it lies inside, INCONCLUSIVE when it straddles the threshold.
// Render through a total Record<Verdict, ...> — never a two-way
// ternary on a boolean, which would render INCONCLUSIVE as FAILS.
export type Verdict = "PASS" | "FAIL" | "INCONCLUSIVE";
export type CheckName =
  | "demographic_parity"
  | "equal_opportunity"
  | "equalized_odds"
  | "disparate_impact";

export interface FairnessCheck {
  value: number | null; // null = not estimable (e.g. a group with no defaulters)
  ciLow: number | null;
  ciHigh: number | null;
  verdict: Verdict;
}

export interface FairnessReport {
  verdict: Verdict;
  checks: Record<CheckName, FairnessCheck>;
  groupMetrics: GroupMetric[];
  // The same audit on a large fresh draw from the generator.
  population: PopulationAudit;
  scoreBands: ScoreBand[];
}

export interface PopulationAudit {
  nSamples: number;
  verdict: Verdict;
  equalizedOdds: FairnessCheck;
  // The group the model mispredicts most: observed vs mean predicted default rate.
  mostMispredicted: {
    group: string;
    defaultRate: number;
    meanPredicted: number;
    gap: number;
    ciLow: number;
    ciHigh: number;
  };
}

export interface GroupMetric {
  group: string;
  nSamples: number;
  nDefaulted: number;
  approvalRate: number | null;
  approvalRateRepaid: number | null; // the equal-opportunity quantity
  defaultRate: number | null;
  avgScore: number | null;
  meanPredicted: number | null;
  calibrationGap: number | null; // observed minus mean predicted default rate
  calibrationLow: number | null; // simultaneous 95% interval over the groups
  calibrationHigh: number | null;
}

/**
 * Human-friendly labels for the engine's attribution players (an input, or a
 * group of inputs that move together). Used to render real SHAP factors
 * returned by the API. Falls back to a title-cased version of the raw name
 * when unmapped.
 */
const API_FEATURE_LABELS: Record<string, string> = {
  fico_score: "FICO Score",
  credit_utilization: "Credit Utilization",
  debt_to_income: "Debt-to-Income",
  paycheck_regularity: "Paycheck Regularity",
  bill_payment_consistency: "Bill Payment Consistency",
  nsf_count: "NSF Incidents",
  overdraft_days: "Overdraft Days",
  rent_payment_score: "Rent Payment History",
  utility_payment_score: "Utility Payment History",
  income_cash_flow_and_balance: "Income, Cash Flow & Balance",
  delinquency_history: "Delinquency History",
  loan_amount: "Loan Amount",
  loan_term_months: "Loan Term",
  employment_years: "Years Employed",
  num_hard_inquiries: "Hard Inquiries",
  credit_age_months: "Credit History Length",
  banking_tenure_months: "Banking Relationship",
  savings_rate: "Savings Rate",
};

/**
 * Category of every attribution player the engine explains over (creditlens/
 * features/engineering.py attribution_players(): each input, with income, cash
 * flows and balance, and delinquency count and recency, grouped), for the waterfall's tag.
 * tests/test_claims.py fails if a player is missing here or a key is not a player.
 */
const API_FEATURE_CATEGORY: Record<string, FeatureContribution["category"]> = {
  fico_score: "core", credit_utilization: "core", debt_to_income: "core",
  delinquency_history: "core", num_hard_inquiries: "core",
  credit_age_months: "core", loan_amount: "core", loan_term_months: "core",
  paycheck_regularity: "behavioral", bill_payment_consistency: "behavioral",
  nsf_count: "behavioral", overdraft_days: "behavioral", employment_years: "behavioral",
  savings_rate: "behavioral",
  income_cash_flow_and_balance: "alternative",
  rent_payment_score: "alternative", utility_payment_score: "alternative",
  banking_tenure_months: "alternative",
};

export function apiFeatureCategory(feature: string): FeatureContribution["category"] {
  return API_FEATURE_CATEGORY[feature] ?? "core";
}

export function apiFeatureLabel(feature: string): string {
  if (API_FEATURE_LABELS[feature]) return API_FEATURE_LABELS[feature];
  return feature
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

/**
 * Shown when GET /fairness/report is unreachable. Nothing in the browser can
 * run an audit, so this is the live engine's own report for its startup model,
 * recorded in engineSnapshot.json: labelled "Demo audit" in the UI, never "Live
 * audit". tests/test_engine_snapshot.py fails if it drifts from the engine.
 */
export function recordedFairnessReport(): FairnessReport {
  return { ...snapshot.fairness, scoreBands: snapshot.score_bands } as FairnessReport;
}

export const RECORDED_ON: string = snapshot.recorded;
