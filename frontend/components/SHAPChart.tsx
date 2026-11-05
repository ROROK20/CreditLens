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

export default function SHAPChart({ contributions, baseline, probability, logOdds }: Props) {
  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.contribution)), 0.1);

  return (
    <div className="bg-panel border border-border rounded-lg p-5">
      <h3 className="text-sm font-semibold mb-1">Feature Contributions</h3>
      <p className="text-xs text-gray-500 mb-4">
        {logOdds
          ? "Each factor's change in log-odds against the average applicant, from the engine's recorded weights (negative = lower risk). The payment burden is drawn under loan amount."
          : "How each factor moves default probability (negative = lower risk, positive = higher risk)"}
      </p>
      {baseline != null && probability != null && contributions.length > 0 && (
        <p className="text-[10px] text-gray-500 -mt-3 mb-4" data-testid="chart-total">
          Baseline {(baseline * 100).toFixed(1)}% → this applicant {(probability * 100).toFixed(1)}%:
          the bars sum to the {((probability - baseline) * 100).toFixed(1)}-point move.
        </p>
      )}
      {contributions.length === 0 && (
        // Reachable on the live path when SHAP attribution is unavailable: say so
        // rather than rendering an empty frame (and never fall back to mock bars).
        <p className="text-xs text-amber-400 mb-2">
          Feature attribution unavailable for this request.
        </p>
      )}
      <div className="space-y-2">
        {contributions.map((c) => {
          const pct = Math.abs(c.contribution) / maxAbs;
          const isPositive = c.contribution < 0;
          return (
            <div key={c.feature} className="flex items-center gap-3 text-xs">
              <div className="w-44 text-right text-gray-300 truncate">{c.label}</div>
              <div className="flex-1 relative h-5 bg-bg/50 rounded overflow-hidden">
                <div className="absolute top-0 left-1/2 w-px h-full bg-gray-500/40" />
                <div
                  className={clsx(
                    "absolute top-0 h-full transition-all duration-300",
                    isPositive ? "bg-accent-green/60 right-1/2" : "bg-accent-red/60 left-1/2",
                  )}
                  style={{ width: `${pct * 50}%` }}
                />
              </div>
              <div
                className={clsx(
                  "w-16 text-right tabular-nums",
                  isPositive ? "text-accent-green" : "text-accent-red",
                )}
              >
                {isPositive ? "−" : "+"}{Math.abs(c.contribution).toFixed(3)}
              </div>
              <div className="w-16 text-left text-gray-500 truncate">
                <span className={clsx(
                  "text-[9px] px-1 py-0.5 rounded",
                  c.category === "core" && "bg-blue-500/20 text-blue-400",
                  c.category === "behavioral" && "bg-purple-500/20 text-purple-400",
                  c.category === "alternative" && "bg-teal-500/20 text-teal-400",
                  c.category === "other" && "bg-gray-500/20 text-gray-400",
                )}>{c.category}</span>
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-4 mt-4 pt-3 border-t border-border text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="w-3 h-2 bg-accent-green/60 rounded-sm" /> Reduces default risk
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-2 bg-accent-red/60 rounded-sm" /> Increases default risk
        </span>
      </div>
    </div>
  );
}
