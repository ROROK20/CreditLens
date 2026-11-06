"use client";

import type { FeatureContribution } from "@/lib/scoring";
import clsx from "clsx";

interface Props {
  contributions: FeatureContribution[];
  // Live attribution only: the baseline and the served probability, which the
  // bars sum between.
  baseline?: number | null;
  probability?: number;
  // No-backend page: the bars are log-odds from the recorded weights, not Shapley values.
  logOdds?: boolean;
}

/**
 * A waterfall: each factor starts where the last one ended, so the claim that the
 * attribution adds up to the served probability is something you can see.
 */
export default function SHAPChart({ contributions, baseline, probability, logOdds }: Props) {
  const start = baseline ?? 0;
  const stops = contributions.reduce<number[]>((acc, c) => [...acc, acc[acc.length - 1] + c.contribution], [start]);
  const lo = Math.min(...stops);
  const hi = Math.max(...stops);
  const x = (v: number) => (hi > lo ? ((v - lo) / (hi - lo)) * 100 : 50);
  const end = stops[stops.length - 1];
  const startLabel = baseline != null ? `Baseline ${(baseline * 100).toFixed(1)}%` : "Average applicant";
  const endLabel = probability != null && baseline != null ? `This applicant ${(probability * 100).toFixed(1)}%` : "This applicant";

  return (
    <div data-testid="contributions">
      <h2 className="flex items-center gap-2.5 border-t-2 border-ink pt-2.5 text-lg font-bold tracking-tight">
        <span className="part-tag">Schedule A</span> Feature Contributions
      </h2>
      <p className="mt-2 max-w-[75ch] text-sm leading-relaxed text-ink-2">
        {logOdds
          ? "Each factor's change in log-odds against the average applicant, from the engine's recorded weights (negative = lower risk). The payment burden is drawn under loan amount."
          : "How each factor moves default probability (negative = lower risk, positive = higher risk)"}
      </p>
      {baseline != null && probability != null && contributions.length > 0 && (
        <p className="mt-1 text-sm font-medium" data-testid="chart-total">
          Baseline {(baseline * 100).toFixed(1)}% → this applicant {(probability * 100).toFixed(1)}%:
          the bars sum to the {((probability - baseline) * 100).toFixed(1)}-point move.
        </p>
      )}
      {contributions.length === 0 && (
        // Reachable on the live path when SHAP attribution is unavailable: say so
        // rather than rendering an empty frame (and never fall back to mock bars).
        <p className="mt-2 text-sm text-caution">Feature attribution unavailable for this request.</p>
      )}
      {contributions.length > 0 && (
        <div className="mt-4 text-sm">
          <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 border-b border-rule-strong pb-1 text-xs text-ink-3 sm:grid-cols-[12.5rem_minmax(0,1fr)_4rem_5.5rem]">
            <div>Factor</div>
            <div className="relative hidden h-4 sm:block">
              <span className="absolute whitespace-nowrap" style={x(start) > 50 ? { right: `${100 - x(start)}%` } : { left: `${x(start)}%` }}>
                {startLabel}
              </span>
            </div>
            <div className="text-right">{logOdds ? "Log-odds" : "Change"}</div>
            <div className="hidden sm:block">Type</div>
          </div>
          {contributions.map((c, i) => {
            const from = stops[i];
            const to = stops[i + 1];
            const left = x(Math.min(from, to));
            const width = Math.abs(x(to) - x(from));
            const lowers = c.contribution < 0;
            return (
              <div
                key={c.feature}
                className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 border-b border-rule py-1.5 sm:grid-cols-[12.5rem_minmax(0,1fr)_4rem_5.5rem] sm:py-1"
              >
                <div data-testid="contribution-label" className="truncate">{c.label}</div>
                <div className="relative order-3 col-span-2 mt-1 h-3.5 sm:order-none sm:col-span-1 sm:mt-0 sm:h-[1.125rem]">
                  <div
                    className={clsx("absolute inset-y-0", lowers ? "bg-lowers" : "bg-raises")}
                    style={{ left: `${left}%`, width: `max(2px, ${width}%)` }}
                  />
                  {i > 0 && <div className="absolute -top-2 bottom-1/2 w-px bg-rule-strong" style={{ left: `${x(from)}%` }} />}
                </div>
                <div
                  data-testid="contribution-value"
                  className={clsx("text-right font-semibold tabular-nums", lowers ? "text-lowers" : "text-raises")}
                >
                  {lowers ? "−" : "+"}{Math.abs(c.contribution).toFixed(3)}
                </div>
                <div className="hidden text-xs text-ink-3 sm:block">{c.category}</div>
              </div>
            );
          })}
          <div className="hidden grid-cols-[12.5rem_minmax(0,1fr)_4rem_5.5rem] gap-x-3 pt-1 text-xs text-ink-3 sm:grid">
            <div />
            <div className="relative h-4">
              <span className="absolute whitespace-nowrap" style={x(end) > 50 ? { right: `${100 - x(end)}%` } : { left: `${x(end)}%` }}>
                {endLabel}
              </span>
            </div>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-3">
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 bg-lowers" /> Reduces default risk</span>
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 bg-raises" /> Increases default risk</span>
          </div>
        </div>
      )}
    </div>
  );
}
