"use client";

import { useMemo } from "react";

interface Props {
  score: number; // 0-1000
  decision: string;
}

export default function ScoreGauge({ score, decision }: Props) {
  const pct = Math.max(0, Math.min(1, score / 1000));
  const angle = pct * 270 - 135; // -135 to +135 degrees

  const color =
    score >= 750 ? "#10b981" :
    score >= 650 ? "#3b82f6" :
    score >= 450 ? "#f59e0b" : "#ef4444";

  const circumference = 2 * Math.PI * 90;
  const dashOffset = useMemo(() => circumference * (1 - pct * 0.75), [pct, circumference]);

  return (
    <div className="relative w-64 h-64 mx-auto">
      <svg viewBox="0 0 200 200" className="w-full h-full -rotate-[135deg]">
        {/* background arc */}
        <circle
          cx="100" cy="100" r="90"
          stroke="#1c2540" strokeWidth="14"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * 0.25}
          strokeLinecap="round"
        />
        {/* score arc */}
        <circle
          cx="100" cy="100" r="90"
          stroke={color} strokeWidth="14"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 0.5s ease, stroke 0.3s ease" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className="text-xs uppercase tracking-wider text-gray-500">Credit Score</div>
        <div data-testid="score-value" className="text-5xl font-bold tabular-nums" style={{ color }}>{score}</div>
        <div className="text-xs text-gray-500 mt-1">out of 1000</div>
        <div
          data-testid="score-decision"
          className="mt-3 px-3 py-1 rounded-md text-xs font-bold tracking-wider"
          style={{ backgroundColor: `${color}20`, color, border: `1px solid ${color}40` }}
        >
          {decision.toUpperCase()}
        </div>
      </div>
    </div>
  );
}
