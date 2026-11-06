import type { ReactNode } from "react";
import { APPROVE_CUTOFF, REVIEW_CUTOFF, type ScoreBand, type ScoringResult } from "@/lib/scoring";

// A total map: a new decision is a compile error, not a band with no colour.
const BAND_COLOUR: Record<ScoringResult["decision"], string> = {
  Approved: "bg-approved",
  Review: "bg-review",
  Denied: "bg-denied",
};

const MAX_SCORE = 1000;

function ScoreRail({ score }: { score: number }) {
  const at = (Math.max(0, Math.min(MAX_SCORE, score)) / MAX_SCORE) * 100;
  const review = (REVIEW_CUTOFF / MAX_SCORE) * 100;
  const approve = (APPROVE_CUTOFF / MAX_SCORE) * 100;
  return (
    <div
      role="img"
      aria-label={`Score ${score} of ${MAX_SCORE}. Denied below ${REVIEW_CUTOFF}, review from ${REVIEW_CUTOFF}, approved from ${APPROVE_CUTOFF}.`}
    >
      <div className="relative">
        <div className="flex h-2.5 gap-px">
          <div className="rail-zone-low" style={{ width: `${review}%` }} />
          <div className="rail-zone-mid" style={{ width: `${approve - review}%` }} />
          <div className="rail-zone-high flex-1" />
        </div>
        {/* Moved with a transform: the wrapper is the rail's width, so a percentage of it is a position on the rail. */}
        <div
          className="absolute -inset-y-2 left-0 w-full transition-transform duration-200 ease-out-quart"
          style={{ transform: `translateX(${at}%)` }}
        >
          <div className="h-full w-1 -translate-x-1/2 bg-band-ink" />
        </div>
      </div>
      <div className="mt-3 flex text-xs leading-4 text-band-ink-2">
        <div style={{ width: `${review}%` }}>
          <div className="tabular-nums">0</div>
          <div>Denied</div>
        </div>
        <div style={{ width: `${approve - review}%` }}>
          <div className="tabular-nums">{REVIEW_CUTOFF}</div>
          <div>Review</div>
        </div>
        <div className="flex flex-1 justify-between">
          <div>
            <div className="tabular-nums">{APPROVE_CUTOFF}</div>
            <div>Approved</div>
          </div>
          <div className="tabular-nums">{MAX_SCORE}</div>
        </div>
      </div>
    </div>
  );
}

interface Props {
  result: ScoringResult;
  band: ScoreBand | null;
  auditedTotal: number;
  observed: string;
  children?: ReactNode;
}

export default function DecisionBand({ result, band, auditedTotal, observed, children }: Props) {
  const pct = `${(result.defaultProbability * 100).toFixed(1)}%`;
  // Says which side it lists: an approval can be net-negative on the largest factor.
  const lowering = result.decision === "Approved";
  const factors = (lowering ? result.topPositives : result.topNegatives).slice(0, 3).map((f) => f.label);

  return (
    <section aria-label="Decision" className={`${BAND_COLOUR[result.decision]} overflow-x-clip text-band-ink`}>
      <div className="mx-auto max-w-[1320px] px-4 pb-7 pt-6 sm:px-6 lg:pb-8 lg:pt-8">
        <div className="grid items-end gap-x-16 gap-y-7 lg:grid-cols-2">
          <div>
            <p className="text-sm font-medium text-band-ink-2">Decision</p>
            <p
              data-testid="score-decision"
              className="mt-1 text-6xl font-extrabold leading-none tracking-tight sm:text-7xl lg:text-8xl"
            >
              {result.decision}
            </p>
            <p className="mt-4 max-w-[60ch] text-base leading-snug">{result.decisionReason}</p>
            {factors.length > 0 && (
              <p className="mt-1 max-w-[60ch] text-base leading-snug text-band-ink-2">
                Largest factors {lowering ? "lowering" : "raising"} risk: {factors.join(", ")}.
              </p>
            )}
          </div>
          <div>
            <p className="text-sm font-medium text-band-ink-2">Credit Score</p>
            <p className="mb-4 mt-1 flex items-baseline gap-2">
              <span
                data-testid="score-value"
                className="text-6xl font-extrabold leading-none tracking-tight tabular-nums"
              >
                {result.creditScore}
              </span>
              <span className="text-sm text-band-ink-2">out of {MAX_SCORE}</span>
            </p>
            <ScoreRail score={result.creditScore} />
          </div>
        </div>

        <dl className="mt-7 grid gap-y-4 border-t border-band-ink-2 pt-4 sm:grid-cols-3 sm:gap-x-8">
          <div>
            <dt className="text-sm text-band-ink-2">Default Probability</dt>
            <dd data-testid="default-probability" className="mt-0.5 text-3xl font-bold tabular-nums">
              {pct}
            </dd>
          </div>
          <div>
            <dt
              className="text-sm text-band-ink-2"
              title="A band of the score. Its observed default rate is measured on the audit set."
            >
              Risk Band · by score
            </dt>
            <dd data-testid="risk-band" className="mt-0.5 text-3xl font-bold">
              {band ? band.label : "n/a"}
            </dd>
          </div>
          <div>
            <dt
              className="text-sm text-band-ink-2"
              title="Of the audited applicants in this score band, the share who defaulted."
            >
              Observed default in this band
            </dt>
            <dd data-testid="band-observed" className="mt-0.5 text-3xl font-bold tabular-nums">
              {observed}
            </dd>
            {band && (
              <dd className="mt-0.5 text-xs text-band-ink-2">
                {band.n} of {auditedTotal} audit applicants, score {band.min_score}–{band.max_score}
              </dd>
            )}
          </div>
        </dl>
        {children}
      </div>
    </section>
  );
}
