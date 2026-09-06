import type { Config } from "tailwindcss";

// Design-token foundation, established Phase 3, expanded Phase 13 (frontend
// visual/UX refinement). The two non-negotiable colour rules are unchanged
// and must never be violated by any future addition here: (1) predicted and
// observed never share a colour: (2) the `intensity` ramp is the only
// semantic storm-severity scale, used identically everywhere it appears.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Three surface levels, not two -- lets a floating panel sit
        // visibly "above" its section without resorting to a heavier
        // border or drop shadow for depth alone.
        bg: { base: "#05070C", surface: "#080B12", elevated: "#0B0F17", overlay: "#10141D" },
        border: {
          subtle: "rgba(255,255,255,0.09)",
          strong: "rgba(255,255,255,0.16)",
          lume: "rgba(120,180,255,0.22)",
        },
        text: { primary: "#F2F5FA", secondary: "#9BA6B8", muted: "#5E6979" },
        accent: { DEFAULT: "#4C8DFF", soft: "#7FB0FF", dim: "#2A4A82" },
        violet: { DEFAULT: "#8B7CF6", soft: "#B4A9FF" },
        amber: { DEFAULT: "#FFB020", soft: "#FFD27A" },
        danger: { DEFAULT: "#F0455C", soft: "#FF8A97" },
        success: { DEFAULT: "#22D3A7", soft: "#7BEBCE" },
        // The ONE semantic colour scale (rule #2 above) -- used identically
        // on the map, in badges, in the classification legend, nowhere else.
        intensity: {
          td: "#4CC9F0", ts: "#4895EF", c1: "#4361EE", c2: "#7209B7",
          c3: "#B5179E", c4: "#F72585", c5: "#FF5400",
        },
        // Rule #1: predicted and observed never share a colour.
        truth: "#22D3A7",
        predicted: "#FFB020",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      boxShadow: {
        glow: "0 0 24px -6px rgba(76,141,255,0.5)",
        "glow-soft": "0 0 40px -12px rgba(76,141,255,0.35)",
        panel: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 12px 32px -16px rgba(0,0,0,0.6)",
        elevated: "0 24px 48px -20px rgba(0,0,0,0.65)",
      },
      backgroundImage: {
        "grid-fine":
          "linear-gradient(rgba(120,180,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(120,180,255,0.05) 1px, transparent 1px)",
        "radial-fade": "radial-gradient(80% 60% at 50% -10%, rgba(76,141,255,0.10), transparent 60%)",
      },
      backgroundSize: {
        grid: "40px 40px",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "pulse-glow": {
          "0%, 100%": { opacity: "0.55" },
          "50%": { opacity: "1" },
        },
        "grid-drift": {
          "0%": { backgroundPosition: "0px 0px" },
          "100%": { backgroundPosition: "40px 40px" },
        },
        "radar-sweep": {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
        "scan-line": {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        "fade-in-up": {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        shimmer: "shimmer 2.2s ease-in-out infinite",
        "pulse-glow": "pulse-glow 3.2s ease-in-out infinite",
        "grid-drift": "grid-drift 18s linear infinite",
        "radar-sweep": "radar-sweep 6s linear infinite",
        "scan-line": "scan-line 4s ease-in-out infinite",
        "fade-in-up": "fade-in-up 0.5s ease-out both",
      },
      transitionTimingFunction: {
        premium: "cubic-bezier(0.16, 1, 0.3, 1)",
      },
    },
  },
  plugins: [],
};

export default config;
