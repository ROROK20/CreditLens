"use client";

import { useState, useMemo, useEffect, useRef } from "react";
import {
  DEFAULT_APPLICANT, scoreApplicant, recordedFairnessReport, apiFeatureLabel, apiFeatureCategory, chartRows,
  bandForScore, INFLOW_TO_MONTHLY_INCOME, OUTFLOW_TO_INFLOW, BALANCE_TO_MONTHLY_INCOME, RECORDED_ON,
  type ApplicantInput, type ScoringResult, type FairnessReport,
  type FeatureContribution, type Verdict, type CheckName,
} from "@/lib/scoring";
import {
  API_URL, checkHealth, scoreApplicantApi, explainApplicantApi, fairnessReportApi, toApiApplicant, ApiRejection,
  type ApiFairnessResponse,
} from "@/lib/api";
import { engineRefusals, engineWarnings } from "@/lib/engineModel.mjs";
import ScoreGauge from "@/components/ScoreGauge";
import SHAPChart from "@/components/SHAPChart";
import {
  Shield, Github, CheckCircle2, XCircle, HelpCircle, Scale, TrendingUp,
  TrendingDown, Wifi, WifiOff,
} from "lucide-react";
import clsx from "clsx";

type EngineMode = "checking" | "live" | "demo";

// One entry per verdict — a TOTAL map, so a new verdict is a compile error
// rather than a silent fall-through to "FAILS".
const VERDICT_COPY: Record<Verdict, { label: string; className: string; Icon: typeof CheckCircle2 }> = {
  PASS: { label: "PASSES", className: "bg-accent-green/20 text-accent-green", Icon: CheckCircle2 },
  FAIL: { label: "FAILS", className: "bg-accent-red/20 text-accent-red", Icon: XCircle },
  INCONCLUSIVE: { label: "INCONCLUSIVE", className: "bg-amber-500/20 text-amber-400", Icon: HelpCircle },
};

// The four gated criteria, in display order, with how each is read.
const CHECK_COPY: Record<CheckName, { label: string; sub: string; threshold: string; ratio?: boolean }> = {
  demographic_parity: { label: "Demographic Parity Δ", sub: "approval rate", threshold: "pass if < 10%" },
  equal_opportunity: { label: "Equal Opportunity Δ", sub: "approval rate among repaid", threshold: "pass if < 5%" },
  equalized_odds: { label: "Equalized Odds Δ", sub: "repaid + defaulted gaps", threshold: "pass if < 10%" },
  disparate_impact: { label: "Disparate Impact", sub: "min / max approval rate", threshold: "0.80 screen", ratio: true },
};
const CHECK_ORDER: CheckName[] = ["demographic_parity", "equal_opportunity", "equalized_odds", "disparate_impact"];

// The form's inputs, every one a slider, each range inside the engine's
// validity domain (creditlens/data/domain.py; tests/test_docs.py checks it):
// outside it the API answers 422 rather than extrapolating.
type NumericKey = {
  [K in keyof ApplicantInput]: ApplicantInput[K] extends number ? K : never;
}[keyof ApplicantInput];

interface FieldDef {
  key: NumericKey;
  label: string;
  type: "number" | "percent";
  min?: number;
  max?: number;
  step?: number;
  // Shown and sent only while this other field is non-zero.
  onlyIf?: NumericKey;
}

const FIELDS: { section: string; fields: FieldDef[] }[] = [
  {
    section: "Income & Employment",
    fields: [
      { key: "annualIncome", label: "Annual Income ($)", type: "number", min: 15000, max: 500000, step: 1000 },
      { key: "employmentYears", label: "Years Employed", type: "number", min: 0, max: 20, step: 0.5 },
      { key: "monthlyInflow", label: "Monthly Inflow ($)", type: "number", min: 900, max: 61000, step: 100 },
      { key: "monthlyOutflow", label: "Monthly Outflow ($)", type: "number", min: 500, max: 50000, step: 100 },
    ],
  },
  {
    section: "Loan Request",
    fields: [
      { key: "loanAmount", label: "Loan Amount ($)", type: "number", min: 1000, max: 49000, step: 500 },
      { key: "loanTermMonths", label: "Term (months)", type: "number", min: 12, max: 60, step: 12 },
    ],
  },
  {
    section: "Credit History",
    fields: [
      { key: "ficoScore", label: "FICO Score", type: "number", min: 350, max: 850, step: 1 },
      { key: "creditUtilization", label: "Credit Utilization", type: "percent", min: 0, max: 0.98, step: 0.01 },
      { key: "debtToIncome", label: "Debt-to-Income", type: "percent", min: 0.01, max: 0.74, step: 0.01 },
      { key: "numDelinquencies", label: "Delinquencies (5yr)", type: "number", min: 0, max: 5, step: 1 },
      { key: "monthsSinceDelinquency", label: "Months Since Last Delinquency", type: "number", min: 0, max: 119, step: 1, onlyIf: "numDelinquencies" },
      { key: "numHardInquiries", label: "Hard Inquiries (1yr)", type: "number", min: 0, max: 8, step: 1 },
      { key: "creditAgeMonths", label: "Credit Age (months)", type: "number", min: 0, max: 360, step: 12 },
    ],
  },
  {
    section: "Behavioral Signals",
    fields: [
      { key: "paycheckRegularity", label: "Paycheck Regularity", type: "percent", min: 0, max: 1, step: 0.01 },
      { key: "billPaymentConsistency", label: "Bill Payment", type: "percent", min: 0.05, max: 1, step: 0.01 },
      { key: "nsfCount", label: "NSF (1yr)", type: "number", min: 0, max: 6, step: 1 },
      { key: "overdraftDays", label: "Overdraft Days (1yr)", type: "number", min: 0, max: 10, step: 1 },
      { key: "rentPaymentScore", label: "Rent Payment", type: "percent", min: 0.13, max: 1, step: 0.01 },
      { key: "utilityPaymentScore", label: "Utility Payment", type: "percent", min: 0.06, max: 1, step: 0.01 },
    ],
  },
  {
    section: "Banking Relationship",
    fields: [
      { key: "avgAccountBalance", label: "Avg Balance ($)", type: "number", min: 500, max: 100000, step: 100 },
      { key: "bankingTenureMonths", label: "Banking Tenure (mo)", type: "number", min: 0, max: 240, step: 6 },
      { key: "savingsRate", label: "Savings Rate", type: "percent", min: 0, max: 0.38, step: 0.01 },
    ],
  },
];

// How often a page that lost its engine asks whether it is back.
const RECONNECT_MS = 10000;

function toFairnessReport(f: ApiFairnessResponse): FairnessReport {
  const groups = Object.entries(f.group_metrics).map(([group, m]) => ({
    group,
    nSamples: m.n_samples,
    nDefaulted: m.n_defaulted,
    approvalRate: m.approval_rate,
    approvalRateRepaid: m.approval_rate_repaid,
    defaultRate: m.default_rate,
    avgScore: m.avg_score,
    meanPredicted: m.mean_predicted_default,
    calibrationGap: m.calibration_in_the_large,
    calibrationLow: m.calibration_ci[0],
    calibrationHigh: m.calibration_ci[1],
  }));
  const checks = Object.fromEntries(
    CHECK_ORDER.map((name) => {
      const c = f.checks[name];
      return [name, { value: c.value, ciLow: c.ci[0], ciHigh: c.ci[1], verdict: c.verdict }];
    }),
  ) as FairnessReport["checks"];
  const odds = f.population.checks.equalized_odds;
  const [worstGroup, worst] = Object.entries(f.population.group_metrics).reduce((a, b) =>
    Math.abs(b[1].calibration_in_the_large ?? 0) > Math.abs(a[1].calibration_in_the_large ?? 0) ? b : a,
  );
  return {
    verdict: f.overall_verdict,
    checks,
    groupMetrics: groups,
    scoreBands: f.score_bands,
    population: {
      nSamples: f.population.n_samples,
      verdict: f.population.overall_verdict,
      equalizedOdds: { value: odds.value, ciLow: odds.ci[0], ciHigh: odds.ci[1], verdict: odds.verdict },
      mostMispredicted: {
        group: worstGroup,
        defaultRate: worst.default_rate ?? 0,
        meanPredicted: worst.mean_predicted_default ?? 0,
        gap: worst.calibration_in_the_large ?? 0,
        ciLow: worst.calibration_ci[0] ?? 0,
        ciHigh: worst.calibration_ci[1] ?? 0,
      },
    },
  };
}

export default function Home() {
  const [applicant, setApplicant] = useState<ApplicantInput>(DEFAULT_APPLICANT);

  // Engine mode: "live" when the real backend answers, "demo" when we fall back
  // to the engine's recorded model. Starts as "checking" until /health responds.
  const [mode, setMode] = useState<EngineMode>("checking");

  // Overrides populated from the REAL engine (POST /score + /explain). When
  // present these replace score/probability/decision/top-factors in the render.
  const [apiResult, setApiResult] = useState<ScoringResult | null>(null);

  // True when the live engine returned a valid score but SHAP attribution was
  // unavailable (explanation_status === "unavailable"). Drives an explicit
  // "explanation unavailable" message instead of silently empty factor lists.
  const [explainUnavailable, setExplainUnavailable] = useState(false);
  // The live attribution's baseline (mean served P(default) over the
  // background): the chart's bars sum from it to the applicant's probability.
  const [attributionBase, setAttributionBase] = useState<number | null>(null);
  const [apiFairness, setApiFairness] = useState<FairnessReport | null>(null);
  // The audit call failed: still nothing to show, and it is asked for again.
  const [fairnessFailed, setFairnessFailed] = useState(false);
  const [fairnessAttempt, setFairnessAttempt] = useState(0);
  // The live engine's 422 reasons when it refuses an applicant (outside its
  // validity domain): shown instead of a score.
  const [rejection, setRejection] = useState<string[] | null>(null);
  // In-domain values few training applicants share (the API's `warnings`).
  const [apiWarnings, setApiWarnings] = useState<string[]>([]);

  // Probe the engine on mount and, while one is configured but not answering, every
  // RECONNECT_MS, so one failed call cannot leave the page on the recording for good. A
  // build with no engine makes no call at all.
  useEffect(() => {
    if (mode === "live") return;
    let cancelled = false;
    const probe = async () => {
      const ok = await checkHealth();
      if (!cancelled) setMode(ok ? "live" : "demo");
    };
    if (mode === "checking") void probe();
    const timer = API_URL === null ? undefined : setInterval(probe, RECONNECT_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [mode]);

  // The live audit: fetched when the engine connects, and asked for again every
  // RECONNECT_MS after a failure and with the next applicant while it is missing. Nothing
  // recorded stands in for it under the live badge.
  const fairnessLoading = useRef(false);
  useEffect(() => {
    if (mode !== "live" || apiFairness !== null || fairnessLoading.current) return;
    fairnessLoading.current = true;
    fairnessReportApi()
      .then((f) => {
        setApiFairness(toFairnessReport(f));
        setFairnessFailed(false);
      })
      .catch(() => {
        setFairnessFailed(true);
        setTimeout(() => setFairnessAttempt((n) => n + 1), RECONNECT_MS);
      })
      .finally(() => {
        fairnessLoading.current = false;
      });
  }, [mode, applicant, apiFairness, fairnessAttempt]);

  // Debounced real-engine scoring whenever the applicant changes (only while
  // live). Any failure flips us back to demo mode and clears everything the engine
  // said, the audit included, until it answers again.
  const reqId = useRef(0);
  useEffect(() => {
    if (mode !== "live") return;
    const myId = ++reqId.current;
    const handle = setTimeout(async () => {
      try {
        const [score, explain] = await Promise.all([
          scoreApplicantApi(applicant),
          explainApplicantApi(applicant, 0),
        ]);
        if (myId !== reqId.current) return; // stale
        const toContrib = (
          f: { feature: string; shap_value: number },
          sign: 1 | -1,
        ): FeatureContribution => ({
          feature: f.feature,
          label: apiFeatureLabel(f.feature),
          // `value` is the applicant's actual feature value; derived/engineered
          // features without a direct input counterpart fall back to 0.
          value:
            (applicant as unknown as Record<string, number>)[f.feature] ?? 0,
          contribution: sign * Math.abs(f.shap_value),
          category: apiFeatureCategory(f.feature),
        });
        const topPositives = explain.top_positive_factors.map((f) => toContrib(f, -1));
        const topNegatives = explain.top_negative_factors.map((f) => toContrib(f, 1));
        // Surface a degraded SHAP explanation explicitly rather than as empty
        // lists: the score is still valid, but the factor breakdown failed.
        setExplainUnavailable(explain.explanation_status === "unavailable");
        setAttributionBase(explain.base_value ?? null);
        setRejection(null);
        setApiWarnings(score.warnings ?? []);
        setApiResult({
          creditScore: score.credit_score,
          defaultProbability: score.default_probability,
          decision: score.decision as ScoringResult["decision"],
          decisionReason: score.decision_reason,
          // The waterfall plots the LIVE SHAP attributions. When attribution is
          // unavailable this is empty and the chart says so.
          // Every feature's attribution, drawn as the largest few plus one
          // "All other factors" row, so the bars sum to the whole move.
          contributions: chartRows(
            Object.entries(explain.shap_values ?? {}).map(([feature, v]) =>
              toContrib({ feature, shap_value: v }, v < 0 ? -1 : 1),
            ),
          ),
          topPositives,
          topNegatives,
        });
      } catch (e) {
        if (myId !== reqId.current) return;
        if (e instanceof ApiRejection) {
          // The engine is up and answered: it will not score this applicant.
          setRejection(e.reasons);
          setApiResult(null);
          setApiWarnings([]);
          return;
        }
        setMode("demo");
        setApiResult(null);
        setExplainUnavailable(false);
        setAttributionBase(null);
        setApiFairness(null);
        setFairnessFailed(false);
        setRejection(null);
        setApiWarnings([]);
      }
    }, 250);
    return () => clearTimeout(handle);
  }, [applicant, mode]);

  // THREE states. Live: the engine's answer, and nothing until it has answered, so no
  // locally computed number can sit under the "Live engine" badge or ship in the
  // prerendered HTML. Demo: the engine's recorded model, run here. Checking: no number at all.
  const recorded = useMemo(() => (mode === "demo" ? scoreApplicant(applicant) : null), [mode, applicant]);
  const result: ScoringResult | null = mode === "live" ? apiResult : recorded;
  // What the recorded model refuses, as the engine would (a live refusal is `rejection`).
  const refusals = mode === "live" ? rejection : mode === "demo" ? engineRefusals(toApiApplicant(applicant)) : [];
  const refused = refusals !== null && refusals.length > 0;
  // What the engine would flag on this applicant, live or by its recorded rules.
  const warnings = mode === "live" ? apiWarnings : mode === "demo" ? engineWarnings(toApiApplicant(applicant)) : [];
  // The fairness card has its OWN liveness: /health can succeed (badge says "Live
  // engine") while /fairness/report is pending or fails, so the card shows no number
  // until the engine answers. Off the engine it is never live: a report arriving late cannot read
  // "Live audit" beside the note saying the audit is a recording.
  const fairnessIsLive = mode === "live" && apiFairness !== null;
  const fairnessPending = mode === "live" && apiFairness === null;
  const fairness: FairnessReport = fairnessIsLive ? apiFairness : recordedFairnessReport();
  // Under the live badge only the live audit's bands: the recording is for the demo.
  const bands = fairnessIsLive || mode === "demo" ? fairness.scoreBands : [];
  const band = result ? bandForScore(result.creditScore, bands) : null;

  const fmtPct = (n: number) => `${(n * 100).toFixed(1)}%`;
  // Undefined audit rates arrive as null: show "n/a", never a fake 0.0%.
  const fmtRate = (n: number | null) => (n === null ? "n/a" : fmtPct(n));
  const fmtCheck = (n: number | null, ratio?: boolean) =>
    n === null ? "n/a" : ratio ? n.toFixed(3) : fmtPct(n);
  // A signed gap in points of default rate.
  const fmtGap = (n: number | null) =>
    n === null ? "n/a" : `${n >= 0 ? "+" : "−"}${Math.abs(n * 100).toFixed(1)}`;
  // While the probe or the first report is in flight the card claims nothing.
  const showVerdicts = mode !== "checking" && !fairnessPending;

  // Inflow and balance are income-linked and outflow inflow-linked (the engine's
  // joint rules, creditlens/data/domain.py), so moving income carries the flows and
  // balance with it, moving inflow carries outflow, and those sliders follow.
  const update = (key: NumericKey, value: number) => {
    if (key === "annualIncome") {
      const scale = value / applicant.annualIncome;
      setApplicant({
        ...applicant,
        annualIncome: value,
        monthlyInflow: Math.round(applicant.monthlyInflow * scale),
        monthlyOutflow: Math.round(applicant.monthlyOutflow * scale),
        avgAccountBalance: Math.round(applicant.avgAccountBalance * scale),
      });
      return;
    }
    if (key === "monthlyInflow") {
      const scale = value / applicant.monthlyInflow;
      setApplicant({ ...applicant, monthlyInflow: value, monthlyOutflow: Math.round(applicant.monthlyOutflow * scale) });
      return;
    }
    setApplicant({ ...applicant, [key]: value });
  };
  const sliderRange = (f: FieldDef) => {
    const monthly = applicant.annualIncome / 12;
    const within = (lo: number, hi: number, base: number) => ({
      min: Math.max(f.min ?? 0, Math.ceil((lo * base) / 100) * 100),
      max: Math.min(f.max ?? Infinity, Math.floor((hi * base) / 100) * 100),
    });
    if (f.key === "monthlyInflow") return within(INFLOW_TO_MONTHLY_INCOME[0], INFLOW_TO_MONTHLY_INCOME[1], monthly);
    if (f.key === "monthlyOutflow") return within(OUTFLOW_TO_INFLOW[0], OUTFLOW_TO_INFLOW[1], applicant.monthlyInflow);
    if (f.key === "avgAccountBalance") return within(BALANCE_TO_MONTHLY_INCOME[0], BALANCE_TO_MONTHLY_INCOME[1], monthly);
    return { min: f.min, max: f.max };
  };

  // Preset scenarios
  const loadPreset = (preset: "strong" | "median" | "weak") => {
    if (preset === "strong") setApplicant({
      ...DEFAULT_APPLICANT, ficoScore: 800, creditUtilization: 0.10, debtToIncome: 0.20,
      numDelinquencies: 0, paycheckRegularity: 0.98, billPaymentConsistency: 0.96, nsfCount: 0,
      overdraftDays: 0, utilityPaymentScore: 0.97, savingsRate: 0.2,
      annualIncome: 120000, monthlyInflow: 10000, monthlyOutflow: 7000,
      employmentYears: 10, avgAccountBalance: 25000,
    });
    // Every input at about its training median; about the 60th percentile of scores.
    if (preset === "median") setApplicant({
      ...DEFAULT_APPLICANT, ficoScore: 677, creditUtilization: 0.31, debtToIncome: 0.26,
      paycheckRegularity: 0.68, billPaymentConsistency: 0.74, overdraftDays: 2,
      rentPaymentScore: 0.77, utilityPaymentScore: 0.73, savingsRate: 0.11,
      annualIncome: 59000, monthlyInflow: 4900, monthlyOutflow: 3500, employmentYears: 3.5,
      avgAccountBalance: 8000, loanAmount: 10000, creditAgeMonths: 36, bankingTenureMonths: 24,
    });
    if (preset === "weak") setApplicant({
      ...DEFAULT_APPLICANT, ficoScore: 580, creditUtilization: 0.75, debtToIncome: 0.55,
      numDelinquencies: 3, monthsSinceDelinquency: 8, paycheckRegularity: 0.60,
      billPaymentConsistency: 0.55, nsfCount: 4, overdraftDays: 10, utilityPaymentScore: 0.5,
      rentPaymentScore: 0.6, savingsRate: 0.01,
      annualIncome: 35000, monthlyInflow: 2900, monthlyOutflow: 2700,
      employmentYears: 1, avgAccountBalance: 1200,
    });
  };

  return (
    <main className="min-h-screen">
      <header className="border-b border-border bg-panel/30 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-gradient-to-br from-teal-500 to-accent-blue rounded-lg flex items-center justify-center glow">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">CreditLens</h1>
              <p className="text-xs text-gray-500">Explainable Credit Scoring</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div
              className={clsx(
                "flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold border",
                mode === "live"
                  ? "border-accent-green/40 text-accent-green bg-accent-green/10"
                  : mode === "demo"
                    ? "border-amber-500/40 text-amber-400 bg-amber-500/10"
                    : "border-border text-gray-400",
              )}
              title={
                mode === "live"
                  ? "Connected to the live CreditLens engine (FastAPI + logistic scorer + SHAP)"
                  : mode === "demo"
                    ? "Backend unreachable — running the engine's recorded model in your browser"
                    : "Checking for the live engine..."
              }
            >
              {mode === "live" ? (
                <Wifi className="w-3.5 h-3.5" />
              ) : (
                <WifiOff className="w-3.5 h-3.5" />
              )}
              {mode === "live" ? "Live engine" : mode === "demo" ? "Demo data" : "Connecting…"}
            </div>
            <a href="https://github.com/ROROK20/CreditLens" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 px-3 py-1.5 text-xs border border-border hover:border-accent-blue rounded-md">
              <Github className="w-3.5 h-3.5" /> View Source
            </a>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {mode === "demo" && (
          <p data-testid="demo-note" className="mb-4 px-3 py-2 text-xs border border-amber-500/40 text-amber-400 bg-amber-500/10 rounded">
            Demo data: no engine is connected. This page runs the engine&apos;s recorded model in your
            browser (seed 42, recorded {RECORDED_ON}): scores match the engine&apos;s, the audit is a
            recording, and the factors are log-odds, not the live engine&apos;s Shapley values.
          </p>
        )}
        {/* Preset buttons */}
        <div className="flex gap-2 mb-4 items-center">
          <span className="text-xs text-gray-500 mr-2">Try preset:</span>
          <button onClick={() => loadPreset("strong")} className="px-3 py-1 text-xs border border-accent-green/40 text-accent-green hover:bg-accent-green/10 rounded">Strong Profile</button>
          <button onClick={() => loadPreset("median")} className="px-3 py-1 text-xs border border-accent-blue/40 text-accent-blue hover:bg-accent-blue/10 rounded">Median Applicant</button>
          <button onClick={() => loadPreset("weak")} className="px-3 py-1 text-xs border border-accent-red/40 text-accent-red hover:bg-accent-red/10 rounded">Weak Profile</button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* LEFT: Form */}
          <div className="lg:col-span-1 space-y-4">
            {FIELDS.map((section) => (
              <div key={section.section} className="bg-panel border border-border rounded-lg p-4">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">
                  {section.section}
                </h3>
                <div className="space-y-3">
                  {section.fields.filter((f) => !f.onlyIf || applicant[f.onlyIf] > 0).map((f) => (
                    <div key={f.key as string}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-400">{f.label}</span>
                        <span className="tabular-nums text-white">
                          {f.type === "percent"
                            ? fmtPct(applicant[f.key])
                            : applicant[f.key].toLocaleString()}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={sliderRange(f).min}
                        max={sliderRange(f).max}
                        step={f.step}
                        value={applicant[f.key]}
                        onChange={(e) => update(f.key, Number(e.target.value))}
                        className="w-full accent-accent-blue"
                      />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* RIGHT: Score + Explanation */}
          <div className="lg:col-span-2 space-y-4">
            {refused ? (
              <div data-testid="live-rejection" className="bg-panel border border-amber-500/40 rounded-lg p-6 flex items-start gap-3">
                <XCircle className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
                <div>
                  <h3 className="text-sm font-semibold text-amber-400 mb-1">
                    Outside the model&apos;s validity domain
                  </h3>
                  <p className="text-xs text-gray-400 leading-relaxed mb-2">
                    {mode === "live" ? "The live engine refuses" : "The engine would refuse"} to score this
                    applicant: the model has no training data here, so any number would be a guess.
                  </p>
                  <ul className="text-xs text-gray-300 space-y-1">
                    {refusals.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : result === null ? (
              <div data-testid="score-pending" className="bg-panel border border-border rounded-lg p-6 text-sm text-gray-400">
                {mode === "live" ? "Waiting for the live engine's answer…" : "Connecting to the scoring engine…"}{" "}
                No score is shown until it answers.
              </div>
            ) : (
            <>
            {/* Score + decision */}
            <div className="bg-panel border border-border rounded-lg p-6 grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
              <ScoreGauge score={result.creditScore} decision={result.decision} />
              <div>
                <div
                  className="text-xs uppercase tracking-wider text-gray-500 mb-1"
                  title="A band of the score. Its observed default rate is measured on the audit set."
                >
                  Risk Band · by score
                </div>
                <div className="text-2xl font-bold mb-3" data-testid="risk-band">{band ? band.label : "—"}</div>

                <div className="text-xs uppercase tracking-wider text-gray-500 mb-1">Default Probability</div>
                <div
                  data-testid="default-probability"
                  className={`text-2xl font-bold tabular-nums mb-3 ${result.defaultProbability >= 0.15 ? "text-accent-red" : "text-gray-100"}`}
                >
                  {fmtPct(result.defaultProbability)}
                </div>

                <div
                  className="text-xs uppercase tracking-wider text-gray-500 mb-1"
                  title="Of the audited applicants in this score band, the share who defaulted."
                >
                  Observed default in this band
                </div>
                <div className="text-2xl font-bold tabular-nums" data-testid="band-observed">
                  {band ? fmtRate(band.observed_default_rate) : "—"}
                </div>
                <div className="text-[10px] text-gray-500 mb-3">
                  {band ? `${band.n} of ${bands.reduce((sum, b) => sum + b.n, 0)} audit applicants, score ${band.min_score}–${band.max_score}` : ""}
                </div>

                <p className="text-xs text-gray-400 leading-relaxed">{result.decisionReason}</p>
                {warnings.length > 0 && (
                  <ul data-testid="domain-warnings" className="mt-2 text-[11px] text-amber-400 space-y-1">
                    {warnings.map((w) => (
                      <li key={w}>{w}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {/* SHAP contributions */}
            <SHAPChart
              contributions={result.contributions}
              baseline={mode === "live" && apiResult ? attributionBase : null}
              probability={result.defaultProbability}
              logOdds={mode === "demo"}
            />

            {/* Top factors — or an explicit "unavailable" state when the live
                engine returned a valid score but SHAP attribution failed. */}
            {explainUnavailable && mode === "live" ? (
              <div className="bg-panel border border-amber-500/40 rounded-lg p-5 flex items-start gap-3">
                <XCircle className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
                <div>
                  <h3 className="text-sm font-semibold text-amber-400 mb-1">
                    Explanation unavailable
                  </h3>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    The score and decision above are valid, but the live engine
                    could not compute the SHAP factor breakdown for this request.
                    Feature-level attribution is temporarily unavailable.
                  </p>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-panel border border-border rounded-lg p-5">
                  <h3 className="text-sm font-semibold flex items-center gap-2 mb-3 text-accent-green">
                    <TrendingDown className="w-4 h-4" /> Reducing Default Risk
                  </h3>
                  <ul className="space-y-1.5">
                    {result.topPositives.map((p) => (
                      <li key={p.feature} className="text-xs flex justify-between">
                        <span className="text-gray-300">{p.label}</span>
                        <span className="tabular-nums text-accent-green">
                          −{Math.abs(p.contribution).toFixed(3)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="bg-panel border border-border rounded-lg p-5">
                  <h3 className="text-sm font-semibold flex items-center gap-2 mb-3 text-accent-red">
                    <TrendingUp className="w-4 h-4" /> Increasing Default Risk
                  </h3>
                  <ul className="space-y-1.5">
                    {result.topNegatives.map((p) => (
                      <li key={p.feature} className="text-xs flex justify-between">
                        <span className="text-gray-300">{p.label}</span>
                        <span className="tabular-nums text-accent-red">
                          +{p.contribution.toFixed(3)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
            </>
            )}

            {/* Fairness Audit */}
            <div className="bg-panel border border-border rounded-lg p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold flex items-center gap-2">
                  <Scale className="w-4 h-4" /> Fairness Audit (Model-Wide)
                  {/* Three states, same rule as the header badge: while the
                      probe is still in flight we claim nothing. */}
                  <span
                    className={clsx(
                      "px-1.5 py-0.5 rounded text-[10px] font-semibold border",
                      mode === "checking"
                        ? "border-border text-gray-400"
                        : fairnessIsLive
                          ? "border-accent-green/40 text-accent-green bg-accent-green/10"
                          : "border-amber-500/40 text-amber-400 bg-amber-500/10",
                    )}
                    title={
                      mode === "checking"
                        ? "Checking for the live fairness audit..."
                        : fairnessIsLive
                          ? "Computed by the live engine over the full test set (GET /fairness/report)"
                          : fairnessPending
                            ? "Waiting for GET /fairness/report; no number is shown until it answers"
                            : `No engine is answering — this is a recorded snapshot of the engine's audit (seed 42, ${RECORDED_ON}), not a live computation`
                    }
                  >
                    {mode === "checking"
                      ? "Checking…"
                      : fairnessIsLive
                        ? "Live audit"
                        : fairnessPending
                          ? "Audit pending"
                          : "Demo audit"}
                  </span>
                </h3>
                {showVerdicts ? (
                  (() => {
                    const v = VERDICT_COPY[fairness.verdict];
                    return (
                      <div
                        className={clsx("flex items-center gap-1.5 px-2 py-1 rounded text-xs font-semibold", v.className)}
                        data-testid="fairness-verdict"
                      >
                        <v.Icon className="w-3.5 h-3.5" />
                        {v.label}
                      </div>
                    );
                  })()
                ) : (
                  <div className="px-2 py-1 rounded text-xs font-semibold border border-border text-gray-400">
                    {fairnessPending ? "Waiting…" : "Checking…"}
                  </div>
                )}
              </div>
              {fairnessPending && (
                <p data-testid="audit-pending" className="text-xs text-gray-400 leading-relaxed">
                  {fairnessFailed
                    ? "The engine did not return its audit. Asking again; no number is shown until it answers."
                    : "Waiting for the engine's audit. The first one covers a large fresh draw, so it can take a moment; no number is shown until it answers."}
                </p>
              )}
              {/* While the probe is in flight the card shows no numbers, so none ship
                  in the prerendered page. */}
              {showVerdicts && (<>
              {/* All FOUR gated criteria, each with its 95% interval and its own
                  verdict. A criterion passes or fails only when its interval
                  clears the threshold; otherwise it is INCONCLUSIVE. */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                {CHECK_ORDER.map((name) => {
                  const copy = CHECK_COPY[name];
                  const c = fairness.checks[name];
                  return (
                    <div key={name} className="metric-card">
                      <div className="text-xs text-gray-500">{copy.label}</div>
                      <div className="text-[10px] text-gray-500">{copy.sub}</div>
                      <div className="text-lg font-bold tabular-nums">{fmtCheck(c.value, copy.ratio)}</div>
                      <div className="text-[10px] text-gray-500 tabular-nums">
                        95% CI {fmtCheck(c.ciLow, copy.ratio)} – {fmtCheck(c.ciHigh, copy.ratio)}
                      </div>
                      <div className="text-[10px] text-gray-500">
                        {copy.threshold}
                        {showVerdicts ? ` · ${c.verdict}` : ""}
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="text-[10px] text-gray-500 mb-3 leading-relaxed">
                Approved = score ≥ 600, the rule <code>/score</code> applies. Intervals are a stratified
                bootstrap, simultaneous over group pairs (conservative for equalized odds); a gate reads PASSES
                or FAILS only when its interval clears the threshold.
              </p>
              <table className="w-full text-xs">
                <thead className="text-gray-500 uppercase tracking-wider">
                  <tr>
                    <th className="text-left py-2">Group</th>
                    <th className="text-right py-2">N</th>
                    <th className="text-right py-2">Defaulted</th>
                    <th className="text-right py-2">Approval</th>
                    <th className="text-right py-2">Approval | Repaid</th>
                    <th className="text-right py-2">Default Rate</th>
                    <th className="text-right py-2">Predicted</th>
                    <th className="text-right py-2" title="Observed minus mean predicted default rate, in points, with its simultaneous 95% interval. Reported, not gated.">Gap (95% CI)</th>
                    <th className="text-right py-2">Avg Score</th>
                  </tr>
                </thead>
                <tbody>
                  {fairness.groupMetrics.map((g) => (
                    <tr key={g.group} className="border-t border-border">
                      <td className="py-2">{g.group}</td>
                      <td className="py-2 text-right tabular-nums">{g.nSamples}</td>
                      <td className="py-2 text-right tabular-nums">{g.nDefaulted}</td>
                      <td className="py-2 text-right tabular-nums">{fmtRate(g.approvalRate)}</td>
                      <td className="py-2 text-right tabular-nums">{fmtRate(g.approvalRateRepaid)}</td>
                      <td className="py-2 text-right tabular-nums">{fmtRate(g.defaultRate)}</td>
                      <td className="py-2 text-right tabular-nums">{fmtRate(g.meanPredicted)}</td>
                      <td className="py-2 text-right tabular-nums" data-testid={`calibration-${g.group}`}>
                        {fmtGap(g.calibrationGap)} ({fmtGap(g.calibrationLow)} to {fmtGap(g.calibrationHigh)})
                      </td>
                      <td className="py-2 text-right tabular-nums">{g.avgScore === null ? "n/a" : g.avgScore}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {/* What the verdict means beyond this one small audit set. */}
              <p data-testid="population-note" className="text-[11px] text-gray-400 mt-3 leading-relaxed">
                The synthetic data injects one real difference: group{" "}
                {fairness.population.mostMispredicted.group} defaults more often at the same features, and no
                feature carries it. On {fairness.population.nSamples.toLocaleString("en-US")} fresh applicants it
                defaults at {fmtPct(fairness.population.mostMispredicted.defaultRate)} against{" "}
                {fmtPct(fairness.population.mostMispredicted.meanPredicted)} predicted (gap{" "}
                {fmtGap(fairness.population.mostMispredicted.gap)} points, 95% CI{" "}
                {fmtGap(fairness.population.mostMispredicted.ciLow)} to{" "}
                {fmtGap(fairness.population.mostMispredicted.ciHigh)}), and equalized odds reads{" "}
                {fmtCheck(fairness.population.equalizedOdds.value)} (95% CI{" "}
                {fmtCheck(fairness.population.equalizedOdds.ciLow)} to{" "}
                {fmtCheck(fairness.population.equalizedOdds.ciHigh)}), {fairness.population.equalizedOdds.verdict} at its
                10% gate, so that gap sits near the gate. Separately, the verdict on this small audit set is mostly
                sample size: a model trained with no injected bias reads equalized odds INCONCLUSIVE at this size too.
              </p>
              </>)}
            </div>
          </div>
        </div>

        <footer className="mt-12 pt-6 border-t border-border text-xs text-gray-500 flex justify-between">
          <p>
            CreditLens · Explainable credit scoring · MIT License ·{" "}
            {/* THREE states, like the header badge. A two-way ternary here made
                "checking" inherit the demo branch, so the production build
                prerendered "Demo model — start the API for live scoring" into
                the served HTML and showed it for ~266ms against a perfectly
                healthy backend. Any claim about the engine has to gate on all
                three states, not just live-vs-not. */}
            {mode === "live"
              ? "Scores from the live engine (logistic scorer + SHAP)"
              : mode === "demo"
                ? "Engine's recorded model — start the API for live scoring"
                : "Connecting to the scoring engine…"}
          </p>
          <p>v0.1.0</p>
        </footer>
      </div>
    </main>
  );
}
