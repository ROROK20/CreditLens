import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        paper: "var(--paper)",
        wash: "var(--wash)",
        ink: "var(--ink)",
        "ink-2": "var(--ink-2)",
        "ink-3": "var(--ink-3)",
        rule: "var(--rule)",
        "rule-strong": "var(--rule-strong)",
        approved: "var(--approved)",
        review: "var(--review)",
        denied: "var(--denied)",
        "band-ink": "var(--band-ink)",
        "band-ink-2": "var(--band-ink-2)",
        lowers: "var(--lowers)",
        raises: "var(--raises)",
        caution: "var(--caution)",
        "caution-wash": "var(--caution-wash)",
      },
      fontFamily: {
        sans: ["var(--font-franklin)", "Franklin Gothic Medium", "Arial", "sans-serif"],
      },
      transitionTimingFunction: {
        "out-quart": "cubic-bezier(0.25, 1, 0.5, 1)",
      },
    },
  },
  plugins: [],
};
export default config;
