"use client";

import { useState, useMemo, useEffect, useRef, type CSSProperties } from "react";
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
import DecisionBand from "@/components/DecisionBand";
import IntervalPlot from "@/components/IntervalPlot";
import SHAPChart from "@/components/SHAPChart";
import { Github } from "lucide-react";
import clsx from "clsx";

type EngineMode = "checking" | "live" | "demo";

// One entry per verdict — a TOTAL map, so a new verdict is a compile error
// rather than a silent fall-through to "FAILS".
const VERDICT_COPY: Record<Verdict, { label: string; className: string }> = {
  PASS: { label: "PASSES", className: "text-lowers" },
  FAIL: { label: "FAILS", className: "text-raises" },
  INCONCLUSIVE: { label: "INCONCLUSIVE", className: "text-caution" },
};

// The four gated criteria, in display order, with how each is read.
const CHECK_COPY: Record<CheckName, { label: string; sub: string; threshold: string; gate: number; ratio?: boolean }> = {
  demographic_parity: { label: "Demographic Parity Δ", sub: "approval rate", threshold: "pass if < 10%", gate: 0.1 },
  equal_opportunity: { label: "Equal Opportunity Δ", sub: "approval rate among repaid", threshold: "pass if < 5%", gate: 0.05 },
  equalized_odds: { label: "Equalized Odds Δ", sub: "repaid + defaulted gaps", threshold: "pass if < 10%", gate: 0.1 },
  disparate_impact: { label: "Disparate Impact", sub: "min / max approval rate", threshold: "0.80 screen", gate: 0.8, ratio: true },
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

const PRESET_BUTTON =
  "border border-rule-strong px-2 py-1.5 text-[0.8125rem] font-medium transition-colors duration-150 ease-out-quart hover:border-ink hover:bg-wash active:bg-rule";

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

  // The sticky header repeats the decision once the band has scrolled away, so the
  // answer stays in view while a line far down the worksheet is being moved.
  const bandRef = useRef<HTMLDivElement>(null);
  const [bandInView, setBandInView] = useState(true);
  useEffect(() => {
    const el = bandRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(([entry]) => setBandInView(entry.isIntersecting));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

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

  // The gate each fairness row is plotted against; the rate gaps share one axis so
  // their intervals can be compared by eye.
  const gapCeiling = Math.ceil(
    Math.max(
      ...CHECK_ORDER.filter((name) => !CHECK_COPY[name].ratio).flatMap((name) => [
        fairness.checks[name].ciHigh ?? 0,
        CHECK_COPY[name].gate,
      ]),
    ) * 20,
  ) / 20;

  return (
    <main className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-rule bg-paper">
        <div className="mx-auto flex max-w-[1320px] items-center gap-3 px-4 py-2.5 sm:gap-4 sm:px-6">
          <div className="flex min-w-0 flex-1 items-baseline gap-3">
            <h1 className="text-lg font-extrabold tracking-tight">CreditLens</h1>
            {!bandInView && result !== null && !refused ? (
              <p aria-hidden className="truncate text-sm tabular-nums">
                <span className="font-bold">{result.decision}</span> · {result.creditScore}
                <span className="hidden sm:inline"> · {fmtPct(result.defaultProbability)} default</span>
              </p>
            ) : (
              <p className="hidden truncate text-sm text-ink-3 sm:block">Explainable Credit Scoring</p>
            )}
          </div>
          <div
            className={clsx(
              "stamp",
              mode === "live" ? "text-lowers" : mode === "demo" ? "text-caution" : "text-ink-3",
            )}
            title={
              mode === "live"
                ? "Connected to the live CreditLens engine (FastAPI + logistic scorer + SHAP)"
                : mode === "demo"
                  ? "Backend unreachable: running the engine's recorded model in your browser"
                  : "Checking for the live engine..."
            }
          >
            {mode === "live" ? "Live engine" : mode === "demo" ? "Demo data" : "Connecting…"}
          </div>
          <a
            href="https://github.com/ROROK20/CreditLens"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="View Source on GitHub"
            className="flex items-center gap-1.5 text-sm font-medium underline decoration-rule-strong underline-offset-4 hover:decoration-ink"
          >
            <Github className="h-4 w-4 sm:hidden" aria-hidden />
            <span className="hidden sm:inline">View Source</span>
          </a>
        </div>
      </header>

      {mode === "demo" && (
        <div className="border-b border-rule bg-caution-wash">
          <p data-testid="demo-note" className="mx-auto max-w-[1320px] px-4 py-2 text-sm leading-snug text-caution sm:px-6">
            Demo data: no engine is connected. This page runs the engine&apos;s recorded model in your
            browser (seed 42, recorded {RECORDED_ON}): scores match the engine&apos;s, the audit is a
            recording, and the factors are log-odds, not the live engine&apos;s Shapley values.
          </p>
        </div>
      )}

      {/* The answer first. Three states, and only one of them has a number. */}
      <div ref={bandRef}>
        {refused ? (
          <section data-testid="live-rejection" className="bg-wash">
            <div className="mx-auto max-w-[1320px] px-4 py-7 sm:px-6 lg:py-9">
              <p className="text-5xl font-extrabold leading-none tracking-tight sm:text-6xl">Not scored</p>
              <h2 className="mt-4 text-lg font-bold text-caution">Outside the model&apos;s validity domain</h2>
              <p className="mt-1 max-w-[70ch] text-sm leading-relaxed text-ink-2">
                {mode === "live" ? "The live engine refuses" : "The engine would refuse"} to score this
                applicant: the model has no training data here, so any number would be a guess.
              </p>
              <ul className="mt-2 max-w-[70ch] list-disc space-y-1 pl-4 text-sm">
                {refusals.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
            </div>
          </section>
        ) : result === null ? (
          <section data-testid="score-pending" className="bg-wash">
            <div className="mx-auto max-w-[1320px] px-4 py-7 sm:px-6 lg:py-9">
              <p className="text-5xl font-extrabold leading-none tracking-tight text-ink-3 sm:text-6xl">No score yet</p>
              <p className="mt-4 max-w-[70ch] text-base text-ink-2">
                {mode === "live" ? "Waiting for the live engine's answer…" : "Connecting to the scoring engine…"}{" "}
                No score is shown until it answers.
              </p>
            </div>
          </section>
        ) : (
          <DecisionBand
            result={result}
            band={band}
            auditedTotal={bands.reduce((sum, b) => sum + b.n, 0)}
            observed={band ? fmtRate(band.observed_default_rate) : "n/a"}
          >
            {warnings.length > 0 && (
              <ul data-testid="domain-warnings" className="mt-4 max-w-[75ch] list-disc space-y-1 pl-4 text-sm">
                {warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            )}
          </DecisionBand>
        )}
      </div>

      <div className="mx-auto grid max-w-[1320px] grid-cols-1 gap-y-12 px-4 py-8 sm:px-6 lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)] lg:gap-x-10">
        {/* The application, as a worksheet */}
        <div className="lg:border-r lg:pr-10">
          <div className="mb-6">
            <p className="text-lg font-bold tracking-tight">The application</p>
            <p className="mt-0.5 text-sm text-ink-2">Move any line; the decision above follows.</p>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className="w-full text-xs text-ink-3">Try preset:</span>
              <button type="button" onClick={() => loadPreset("strong")} className={PRESET_BUTTON}>Strong Profile</button>
              <button type="button" onClick={() => loadPreset("median")} className={PRESET_BUTTON}>Median Applicant</button>
              <button type="button" onClick={() => loadPreset("weak")} className={PRESET_BUTTON}>Weak Profile</button>
            </div>
          </div>
          <div className="space-y-7">
            {FIELDS.map((section, part) => (
              <section key={section.section} aria-labelledby={`part-${part + 1}`}>
                <h2 id={`part-${part + 1}`} className="flex items-center gap-2.5 border-t-2 border-ink pt-2 text-sm font-bold">
                  <span className="part-tag">Part {part + 1}</span> {section.section}
                </h2>
                <div className="mt-1">
                  {section.fields.filter((f) => !f.onlyIf || applicant[f.onlyIf] > 0).map((f, line) => {
                    const id = `line-${f.key}`;
                    const range = sliderRange(f);
                    const shown = f.type === "percent" ? fmtPct(applicant[f.key]) : applicant[f.key].toLocaleString("en-US");
                    const lo = range.min ?? 0;
                    const hi = range.max ?? 1;
                    const fill = hi > lo ? ((applicant[f.key] - lo) / (hi - lo)) * 100 : 0;
                    return (
                      <div key={f.key as string} className="pb-0.5 pt-2">
                        <div className="flex items-baseline justify-between gap-3 text-sm">
                          <label htmlFor={id} className="min-w-0">
                            <span className="line-no" aria-hidden>{part + 1}{"abcdefgh"[line]}</span>
                            {f.label}
                          </label>
                          <output htmlFor={id} className="font-semibold tabular-nums">{shown}</output>
                        </div>
                        <input
                          id={id}
                          type="range"
                          min={range.min}
                          max={range.max}
                          step={f.step}
                          value={applicant[f.key]}
                          aria-valuetext={shown}
                          onChange={(e) => update(f.key, Number(e.target.value))}
                          className="fader"
                          style={{ "--fill": `${fill}%` } as CSSProperties}
                        />
                      </div>
                    );
                  })}
                </div>
              </section>
            ))}
          </div>
        </div>

        {/* Why, and whether the model can be trusted to say so */}
        <div className="min-w-0 space-y-12">
          {result !== null && !refused && (
            <div>
              <SHAPChart
                contributions={result.contributions}
                baseline={mode === "live" && apiResult ? attributionBase : null}
                probability={result.defaultProbability}
                logOdds={mode === "demo"}
              />

              {/* Top factors, or an explicit "unavailable" state when the live
                  engine returned a valid score but SHAP attribution failed. */}
              {explainUnavailable && mode === "live" ? (
                <div className="mt-6 border border-caution bg-caution-wash px-4 py-3">
                  <h3 className="text-sm font-bold text-caution">Explanation unavailable</h3>
                  <p className="mt-1 text-sm leading-relaxed text-ink-2">
                    The score and decision above are valid, but the live engine
                    could not compute the SHAP factor breakdown for this request.
                    Feature-level attribution is temporarily unavailable.
                  </p>
                </div>
              ) : (
                <div className="mt-7 grid gap-x-10 gap-y-6 md:grid-cols-2">
                  <div>
                    <h3 className="border-b border-rule-strong pb-1 text-sm font-bold text-lowers">Reducing Default Risk</h3>
                    <ul className="divide-y divide-rule">
                      {result.topPositives.map((p) => (
                        <li key={p.feature} className="flex justify-between gap-3 py-1 text-sm">
                          <span>{p.label}</span>
                          <span className="tabular-nums text-lowers">−{Math.abs(p.contribution).toFixed(3)}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h3 className="border-b border-rule-strong pb-1 text-sm font-bold text-raises">Increasing Default Risk</h3>
                    <ul className="divide-y divide-rule">
                      {result.topNegatives.map((p) => (
                        <li key={p.feature} className="flex justify-between gap-3 py-1 text-sm">
                          <span>{p.label}</span>
                          <span className="tabular-nums text-raises">+{p.contribution.toFixed(3)}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Fairness Audit */}
          <section aria-labelledby="schedule-b">
            <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-t-2 border-ink pt-2.5">
              <h2 id="schedule-b" className="flex flex-wrap items-center gap-2.5 text-lg font-bold tracking-tight">
                <span className="part-tag">Schedule B</span> Fairness Audit (Model-Wide)
                {/* Three states, same rule as the header badge: while the
                    probe is still in flight we claim nothing. */}
                <span
                  className={clsx(
                    "stamp",
                    mode === "checking" ? "text-ink-3" : fairnessIsLive ? "text-lowers" : "text-caution",
                  )}
                  title={
                    mode === "checking"
                      ? "Checking for the live fairness audit..."
                      : fairnessIsLive
                        ? "Computed by the live engine over the full test set (GET /fairness/report)"
                        : fairnessPending
                          ? "Waiting for GET /fairness/report; no number is shown until it answers"
                          : `No engine is answering: this is a recorded snapshot of the engine's audit (seed 42, ${RECORDED_ON}), not a live computation`
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
              </h2>
              {showVerdicts ? (
                (() => {
                  const v = VERDICT_COPY[fairness.verdict];
                  return (
                    <div className={clsx("stamp text-sm font-bold", v.className)} data-testid="fairness-verdict">
                      {v.label}
                    </div>
                  );
                })()
              ) : (
                <div className="stamp text-ink-3">{fairnessPending ? "Waiting…" : "Checking…"}</div>
              )}
            </div>
            {fairnessPending && (
              <p data-testid="audit-pending" className="mt-3 max-w-[70ch] text-sm leading-relaxed text-ink-2">
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
            <div className="mt-3 divide-y divide-rule border-b border-rule">
              {CHECK_ORDER.map((name) => {
                const copy = CHECK_COPY[name];
                const c = fairness.checks[name];
                return (
                  <div key={name} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 py-2.5 text-sm md:grid-cols-[13rem_minmax(0,1fr)_3.5rem_11rem]">
                    <div>
                      <div className="font-semibold">{copy.label}</div>
                      <div className="text-xs text-ink-3">{copy.sub}</div>
                    </div>
                    <div className="order-3 col-span-2 md:order-none md:col-span-1">
                      <IntervalPlot
                        value={c.value}
                        low={c.ciLow}
                        high={c.ciHigh}
                        gate={copy.gate}
                        passesBelow={!copy.ratio}
                        domain={copy.ratio ? [Math.min(0.5, c.ciLow ?? 0.5), 1] : [0, gapCeiling]}
                        label={`${copy.label}: ${fmtCheck(c.value, copy.ratio)}, 95% interval ${fmtCheck(c.ciLow, copy.ratio)} to ${fmtCheck(c.ciHigh, copy.ratio)}, ${copy.threshold}, ${c.verdict}`}
                      />
                    </div>
                    <div className="text-right text-base font-bold tabular-nums">{fmtCheck(c.value, copy.ratio)}</div>
                    <div className="order-4 col-span-2 text-xs leading-snug text-ink-2 md:order-none md:col-span-1">
                      <div className="tabular-nums">
                        95% CI {fmtCheck(c.ciLow, copy.ratio)} – {fmtCheck(c.ciHigh, copy.ratio)}
                      </div>
                      <div>
                        {copy.threshold} · <span className={clsx("font-semibold", VERDICT_COPY[c.verdict].className)}>{c.verdict}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="mt-3 max-w-[85ch] text-xs leading-relaxed text-ink-2">
              Shaded: inside the gate. Line: the 95% interval. Dot: the reading.
              Approved = score ≥ 600, the rule <code>/score</code> applies. Intervals are a stratified
              bootstrap, simultaneous over group pairs (conservative for equalized odds); a gate reads PASSES
              or FAILS only when its interval clears the threshold.
            </p>
            <div className="mt-5 overflow-x-auto" tabIndex={0} role="region" aria-label="Group metrics table">
              <table className="w-full min-w-[46rem] text-sm">
                <thead className="text-xs text-ink-3">
                  <tr className="border-b border-rule-strong">
                    <th className="py-1.5 pr-3 text-left font-medium">Group</th>
                    <th className="px-2 py-1.5 text-right font-medium">N</th>
                    <th className="px-2 py-1.5 text-right font-medium">Defaulted</th>
                    <th className="px-2 py-1.5 text-right font-medium">Approval</th>
                    <th className="px-2 py-1.5 text-right font-medium">Approval | Repaid</th>
                    <th className="px-2 py-1.5 text-right font-medium">Default Rate</th>
                    <th className="px-2 py-1.5 text-right font-medium">Predicted</th>
                    <th className="px-2 py-1.5 text-right font-medium" title="Observed minus mean predicted default rate, in points, with its simultaneous 95% interval. Reported, not gated.">Gap (95% CI)</th>
                    <th className="py-1.5 pl-2 text-right font-medium">Avg Score</th>
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {fairness.groupMetrics.map((g) => (
                    <tr key={g.group} className="border-b border-rule">
                      <td className="py-1.5 pr-3 font-semibold">{g.group}</td>
                      <td className="px-2 py-1.5 text-right">{g.nSamples}</td>
                      <td className="px-2 py-1.5 text-right">{g.nDefaulted}</td>
                      <td className="px-2 py-1.5 text-right">{fmtRate(g.approvalRate)}</td>
                      <td className="px-2 py-1.5 text-right">{fmtRate(g.approvalRateRepaid)}</td>
                      <td className="px-2 py-1.5 text-right">{fmtRate(g.defaultRate)}</td>
                      <td className="px-2 py-1.5 text-right">{fmtRate(g.meanPredicted)}</td>
                      <td className="whitespace-nowrap px-2 py-1.5 text-right" data-testid={`calibration-${g.group}`}>
                        {fmtGap(g.calibrationGap)} ({fmtGap(g.calibrationLow)} to {fmtGap(g.calibrationHigh)})
                      </td>
                      <td className="py-1.5 pl-2 text-right">{g.avgScore === null ? "n/a" : g.avgScore}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {/* What the verdict means beyond this one small audit set. */}
            <p data-testid="population-note" className="mt-4 max-w-[75ch] text-sm leading-relaxed text-ink-2">
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
          </section>
        </div>
      </div>

      <footer className="mx-auto mt-4 flex max-w-[1320px] flex-wrap justify-between gap-x-6 gap-y-1 border-t border-rule px-4 py-5 text-xs text-ink-3 sm:px-6">
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
              ? "Engine's recorded model. Start the API for live scoring"
              : "Connecting to the scoring engine…"}
        </p>
        <p>v0.1.0</p>
      </footer>
    </main>
  );
}
