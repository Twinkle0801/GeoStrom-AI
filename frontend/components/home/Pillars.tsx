"use client";

import { motion } from "framer-motion";
import GlassPanel from "@/components/ui/GlassPanel";
import SectionHeader from "@/components/ui/SectionHeader";
import { fadeUp, reducedMotionVariants, staggerContainer, usePrefersReducedMotion } from "@/lib/motion";

const PILLARS = [
  {
    title: "Observed evidence",
    body: "Historical IBTrACS positions, wind, and pressure — real measurements, never model output.",
  },
  {
    title: "Model predictions",
    body: "Baseline and exploratory models forecast intensity and track from that history, retrospectively evaluated on held-out storms.",
  },
  {
    title: "Satellite structure",
    body: "HURSAT-B1 imagery and ADT-derived Dvorak scene labels describe the storm's visual structure where available.",
  },
  {
    title: "Evidence-grounded explanation",
    body: "Gemini narrates the stored evidence in plain language — it never computes a number, and every claim is checked against the record.",
  },
] as const;

export default function Pillars() {
  const reducedMotion = usePrefersReducedMotion();
  return (
    <section className="mx-auto max-w-7xl px-6 py-16">
      <SectionHeader
        eyebrow="What GeoStrom AI shows"
        title="Four kinds of information — never blended"
        description="Every panel in this product states plainly whether you're looking at an observation, a prediction, satellite-derived structure, or an AI-generated explanation."
      />
      <motion.div
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-80px" }}
        variants={reducedMotion ? reducedMotionVariants : staggerContainer}
        className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
      >
        {PILLARS.map((p) => (
          <motion.div key={p.title} variants={reducedMotion ? reducedMotionVariants : fadeUp}>
            <GlassPanel className="h-full p-5">
              <h3 className="text-sm font-semibold text-text-primary">{p.title}</h3>
              <p className="mt-2 text-sm text-text-secondary">{p.body}</p>
            </GlassPanel>
          </motion.div>
        ))}
      </motion.div>
    </section>
  );
}
