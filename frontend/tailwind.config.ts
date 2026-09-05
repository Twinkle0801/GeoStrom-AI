import type { Config } from "tailwindcss";

// Phase 3 scope note (docs/UI_UX_ARCHITECTURE.md §1.2): this establishes the
// design-token FOUNDATION only -- dark ground, type scale, the two
// non-negotiable colour rules (predicted != observed; intensity ramp is the
// only semantic scale). Cinematic motion, glassmorphism refinement, and the
// 3D globe belong to a later frontend phase, not this vertical slice.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: { base: "#05070C", elevated: "#0B0F17" },
        border: { subtle: "rgba(255,255,255,0.09)", lume: "rgba(120,180,255,0.22)" },
        text: { primary: "#F2F5FA", secondary: "#9BA6B8", muted: "#5E6979" },
        accent: { DEFAULT: "#4C8DFF", soft: "#7FB0FF" },
        // The ONE semantic colour scale (UI_UX_ARCHITECTURE.md rule #2) --
        // used identically on the map, in badges, and nowhere else.
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
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
