/**
 * Typed client for the CreditLens FastAPI backend.
 *
 * The base URL is NEXT_PUBLIC_API_URL, inlined at build time (lib/apiUrl.mjs):
 * under `next dev` it defaults to http://localhost:8002; a production build
 * without it has no engine and makes no network call. Every call is best-effort:
 * on ANY error (no engine, network down, non-2xx, malformed body) the function
 * rejects, and the CALLER is expected to fall back to the engine's recorded model
 * in `lib/engineModel.mjs`. This keeps `npm run build` and a no-backend static deploy
 * fully functional.
 */

import { resolveApiUrl } from "./apiUrl.mjs";
import type { ApplicantInput, CheckName, ScoreBand, Verdict } from "./scoring";

export const API_URL = resolveApiUrl(process.env.NEXT_PUBLIC_API_URL, process.env.NODE_ENV);

/** Shape the backend's ApplicantFeatures pydantic model expects. */
interface ApiApplicant {
  annual_income: number;
  loan_amount: number;
  loan_term_months: number;
  debt_to_income: number;
  employment_years: number;
  fico_score: number;
  credit_utilization: number;
  num_delinquencies: number;
  months_since_delinquency: number;
  num_hard_inquiries: number;
  credit_age_months: number;
  monthly_inflow: number;
  monthly_outflow: number;
  paycheck_regularity: number;
  bill_payment_consistency: number;
  nsf_count: number;
  overdraft_days: number;
  rent_payment_score: number;
  utility_payment_score: number;
  banking_tenure_months: number;
  avg_account_balance: number;
  savings_rate: number;
}

export interface ApiScoreResponse {
  credit_score: number;
  default_probability: number;
  decision: string;
  decision_reason: string;
  // In-domain values few training applicants share (outside the central 99%).
  warnings: string[];
}

/** The engine answered 422: it refuses this applicant (outside its validity
 *  domain, or contradictory fields). Not an outage — the caller stays live. */
export class ApiRejection extends Error {
  constructor(public reasons: string[]) {
    super(reasons.join("; "));
    this.name = "ApiRejection";
  }
}

const rejectionReasons = (body: unknown): string[] => {
  const detail = (body as { detail?: unknown })?.detail;
  if (!Array.isArray(detail)) return ["The engine rejected this applicant."];
  return detail.map((d: { loc?: unknown[]; msg?: string }) => {
    const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : undefined;
    const msg = (d.msg ?? "invalid").replace(/^Value error, /, "");
    return typeof field === "string" && field !== "body" ? `${field}: ${msg}` : msg;
  });
};

export interface ApiShapFactor {
  feature: string;
  shap_value: number;
  impact: string;
}

export interface ApiExplainResponse {
  applicant_id: number;
  credit_score: number;
  default_probability: number;
  decision: string;
  explanation_text: string;
  top_positive_factors: ApiShapFactor[];
  top_negative_factors: ApiShapFactor[];
  // The full attribution, in P(default): base_value + sum(shap_values) is the
  // served probability. Null / empty when attribution is unavailable.
  base_value: number | null;
  shap_values: Record<string, number>;
  attribution_method?: string;
  // "ok" when SHAP attribution succeeded; "unavailable" when it failed (the
  // score/decision are still valid but the factor lists come back empty).
  // Optional for backward compatibility with an older backend that omits it.
  explanation_status?: "ok" | "unavailable";
  explanation_status_reason?: string | null;
}

// Undefined rates (e.g. a group with no defaulters) arrive as null — JSON has
// no NaN. Every rate below may be null; render it as "n/a", never as 0.
export interface ApiGroupMetric {
  n_samples: number;
  n_repaid: number;
  n_defaulted: number;
  n_approved: number;
  approval_rate: number | null;
  approval_rate_repaid: number | null;
  approval_rate_defaulted: number | null;
  avg_score: number | null;
  default_rate: number | null;
  mean_predicted_default: number | null;
  calibration_in_the_large: number | null;
  calibration_ci: [number | null, number | null];
}

export interface ApiFairnessCheck {
  definition: string;
  value: number | null;
  ci: [number | null, number | null];
  threshold: number;
  passes_when: string;
  verdict: Verdict;
  worst_pair: [string, string] | null;
}

export interface ApiFairnessResponse {
  // Three states (creditlens/fairness/metrics.py): FAIL if any criterion's
  // interval lies beyond its threshold, PASS if every one lies inside, else
  // INCONCLUSIVE. overall_pass is true only for PASS — never render it as a
  // two-way PASSES/FAILS switch, or INCONCLUSIVE reads as FAILS.
  overall_verdict: Verdict;
  overall_pass: boolean;
  decision_rule: string;
  confidence: number;
  checks: Record<CheckName, ApiFairnessCheck>;
  recommendations: string[];
  group_metrics: Record<string, ApiGroupMetric>;
  // Each risk band's observed default rate on this audit set.
  score_bands: ScoreBand[];
  // The same audit of the served model on a large fresh draw.
  population: {
    n_samples: number;
    overall_verdict: Verdict;
    checks: Record<CheckName, ApiFairnessCheck>;
    group_metrics: Record<string, ApiGroupMetric>;
  };
}

/** Map the frontend camelCase applicant onto the backend's snake_case schema. */
export function toApiApplicant(a: ApplicantInput): ApiApplicant {
  return {
    annual_income: a.annualIncome,
    loan_amount: a.loanAmount,
    loan_term_months: a.loanTermMonths,
    debt_to_income: a.debtToIncome,
    employment_years: a.employmentYears,
    fico_score: a.ficoScore,
    credit_utilization: a.creditUtilization,
    num_delinquencies: a.numDelinquencies,
    months_since_delinquency: a.numDelinquencies > 0 ? a.monthsSinceDelinquency : -1,
    num_hard_inquiries: a.numHardInquiries,
    credit_age_months: a.creditAgeMonths,
    monthly_inflow: a.monthlyInflow,
    monthly_outflow: a.monthlyOutflow,
    paycheck_regularity: a.paycheckRegularity,
    bill_payment_consistency: a.billPaymentConsistency,
    nsf_count: a.nsfCount,
    overdraft_days: a.overdraftDays,
    rent_payment_score: a.rentPaymentScore,
    utility_payment_score: a.utilityPaymentScore,
    banking_tenure_months: a.bankingTenureMonths,
    avg_account_balance: a.avgAccountBalance,
    savings_rate: a.savingsRate,
  };
}

const DEFAULT_TIMEOUT_MS = 6000;
// The first report after a start computes the large-draw audit (seconds, more on a
// small host); every later one is cached.
const FAIRNESS_TIMEOUT_MS = 30000;

async function fetchJson<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  if (API_URL === null) throw new Error("No engine configured for this build.");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
    if (res.status === 422) {
      throw new ApiRejection(rejectionReasons(await res.json().catch(() => null)));
    }
    if (!res.ok) {
      throw new Error(`API ${path} returned ${res.status}`);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

/** GET /health — used to detect whether the real engine is reachable. */
export async function checkHealth(timeoutMs = 2500): Promise<boolean> {
  try {
    const h = await fetchJson<{ status: string }>("/health", undefined, timeoutMs);
    return h.status === "ok";
  } catch {
    return false;
  }
}

/** POST /score — real engine scoring. Throws on any failure (caller falls back). */
export async function scoreApplicantApi(
  a: ApplicantInput,
): Promise<ApiScoreResponse> {
  return fetchJson<ApiScoreResponse>("/score", {
    method: "POST",
    body: JSON.stringify(toApiApplicant(a)),
  });
}

/** POST /explain/{idx} — real SHAP explanation. Throws on any failure. */
export async function explainApplicantApi(
  a: ApplicantInput,
  idx = 0,
): Promise<ApiExplainResponse> {
  return fetchJson<ApiExplainResponse>(`/explain/${idx}`, {
    method: "POST",
    body: JSON.stringify(toApiApplicant(a)),
  });
}

/** GET /fairness/report — real fairness audit. Throws on any failure. */
export async function fairnessReportApi(): Promise<ApiFairnessResponse> {
  return fetchJson<ApiFairnessResponse>("/fairness/report", undefined, FAIRNESS_TIMEOUT_MS);
}
