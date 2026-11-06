interface Props {
  value: number | null;
  low: number | null;
  high: number | null;
  gate: number;
  // Which side of the gate passes: the rate gaps pass below it, the ratio above.
  passesBelow: boolean;
  domain: [number, number];
  label: string;
}

/**
 * One fairness gate on its own axis: the shaded side passes, the line is the 95%
 * interval, the dot is the reading. A line that crosses the gate is why a verdict
 * reads INCONCLUSIVE, and this makes that visible instead of asserting it.
 */
export default function IntervalPlot({ value, low, high, gate, passesBelow, domain, label }: Props) {
  if (value === null || low === null || high === null) {
    return <div className="h-6 border-b border-rule" aria-hidden />;
  }
  const x = (v: number) => Math.max(0, Math.min(1, (v - domain[0]) / (domain[1] - domain[0]))) * 100;
  const passLeft = passesBelow ? 0 : x(gate);
  const passWidth = passesBelow ? x(gate) : 100 - x(gate);
  return (
    <div className="relative h-6" role="img" aria-label={label}>
      <div className="absolute inset-y-0 bg-wash" style={{ left: `${passLeft}%`, width: `${passWidth}%` }} />
      <div className="absolute inset-y-0 w-px bg-rule-strong" style={{ left: `${x(gate)}%` }} />
      <div
        className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-ink"
        style={{ left: `${x(low)}%`, width: `${x(high) - x(low)}%` }}
      />
      <div className="absolute top-1/2 h-2 w-0.5 -translate-y-1/2 bg-ink" style={{ left: `${x(low)}%` }} />
      <div className="absolute top-1/2 h-2 w-0.5 -translate-x-full -translate-y-1/2 bg-ink" style={{ left: `${x(high)}%` }} />
      <div
        className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-paper bg-ink"
        style={{ left: `${x(value)}%` }}
      />
    </div>
  );
}
