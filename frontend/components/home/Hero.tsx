"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import HeroGlobeClient from "@/components/globe/HeroGlobeClient";
import { fadeUp, reducedMotionVariants, staggerContainer, usePrefersReducedMotion } from "@/lib/motion";

export default function Hero() {
  const reducedMotion = usePrefersReducedMotion();
  const item = reducedMotion ? reducedMotionVariants : fadeUp;

  return (
    <section className="relative mx-auto grid max-w-7xl grid-cols-1 items-center gap-12 px-6 py-20 sm:py-28 lg:grid-cols-2">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={reducedMotion ? reducedMotionVariants : staggerContainer}
      >
        <motion.div
          variants={item}
          className="text-xs font-semibold uppercase tracking-[0.16em] text-accent-soft"
        >
          Retrospective cyclone intelligence
        </motion.div>
        <motion.h1
          variants={item}
          className="mt-4 text-4xl font-semibold leading-[1.05] tracking-tight text-text-primary sm:text-5xl lg:text-6xl"
        >
          Understand the structure.
          <br />
          Predict the trajectory.
        </motion.h1>
        <motion.p variants={item} className="mt-6 max-w-xl text-base text-text-secondary sm:text-lg">
          GeoStrom AI uses historical satellite observations and machine learning to study tropical
          cyclone structure, intensity, and track behaviour across the North Atlantic basin.
        </motion.p>
        <motion.p variants={item} className="mt-2 max-w-xl text-sm text-text-muted">
          A retrospective research platform — every prediction shown is historical baseline model
          output, evaluated against known outcomes. Not an operational forecasting system.
        </motion.p>
        <motion.div variants={item} className="mt-8 flex flex-wrap gap-3">
          <Link
            href="/storms"
            className="rounded-lg bg-accent px-5 py-2.5 text-sm font-semibold text-white shadow-[0_0_24px_-6px_rgba(76,141,255,0.65)] transition-transform hover:scale-[1.02] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-soft"
          >
            Explore Storms
          </Link>
          <Link
            href="/methodology"
            className="rounded-lg border border-border-subtle px-5 py-2.5 text-sm font-semibold text-text-primary transition-colors hover:bg-white/5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-soft"
          >
            Explore Methodology
          </Link>
        </motion.div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, scale: reducedMotion ? 1 : 0.94 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: reducedMotion ? 0 : 0.6, ease: "easeOut" }}
        className="flex items-center justify-center"
      >
        <HeroGlobeClient />
      </motion.div>
    </section>
  );
}
