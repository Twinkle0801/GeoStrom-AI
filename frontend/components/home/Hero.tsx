"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import HeroGlobeClient from "@/components/globe/HeroGlobeClient";
import { ArrowRightIcon } from "@/components/ui/Icons";
import { fadeUp, reducedMotionVariants, staggerContainer, usePrefersReducedMotion } from "@/lib/motion";

const CAPABILITY_STRIP = [
  "Historical satellite intelligence", "Pattern classification", "Intensity prediction", "Track prediction",
] as const;

export default function Hero() {
  const reducedMotion = usePrefersReducedMotion();
  const item = reducedMotion ? reducedMotionVariants : fadeUp;

  return (
    <section className="relative overflow-hidden">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[640px] bg-[radial-gradient(70%_50%_at_50%_0%,rgba(76,141,255,0.14),transparent_65%)]"
      />
      <div className="relative mx-auto grid max-w-7xl grid-cols-1 items-center gap-12 px-6 py-20 sm:py-28 lg:grid-cols-2">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={reducedMotion ? reducedMotionVariants : staggerContainer}
        >
          <motion.div
            variants={item}
            className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-accent-soft"
          >
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent-soft/60 motion-reduce:hidden" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-accent-soft" />
            </span>
            AI-powered tropical cyclone intelligence
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

          <motion.div variants={item} className="mt-5 flex flex-wrap gap-x-4 gap-y-1.5">
            {CAPABILITY_STRIP.map((c, i) => (
              <span key={c} className="flex items-center gap-1.5 text-xs text-text-secondary">
                {i > 0 && <span aria-hidden className="h-1 w-1 rounded-full bg-text-muted/50" />}
                {c}
              </span>
            ))}
          </motion.div>

          <motion.div variants={item} className="mt-8 flex flex-wrap gap-3">
            <Link
              href="/storms"
              className="group inline-flex items-center gap-1.5 rounded-lg bg-accent px-5 py-2.5 text-sm font-semibold text-white shadow-glow transition-all duration-300 ease-premium hover:scale-[1.02] hover:shadow-glow-soft focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-soft"
            >
              Explore Storms
              <ArrowRightIcon width={14} height={14} className="transition-transform duration-300 group-hover:translate-x-0.5" />
            </Link>
            <Link
              href="/methodology"
              className="rounded-lg border border-border-subtle px-5 py-2.5 text-sm font-semibold text-text-primary transition-colors hover:border-border-strong hover:bg-white/5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-soft"
            >
              Explore Methodology
            </Link>
          </motion.div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: reducedMotion ? 1 : 0.94 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: reducedMotion ? 0 : 0.6, ease: "easeOut" }}
          className="relative flex items-center justify-center"
        >
          {/* Purely decorative atmospheric rings behind the globe -- never
              presented as a real radar/orbit visualization, per task §7/§18's
              "decorative animation is allowed only if clearly decorative". */}
          <div aria-hidden className="pointer-events-none absolute h-[560px] w-[560px] max-w-full">
            <div className="absolute inset-0 rounded-full border border-accent/10" />
            <div className="absolute inset-8 rounded-full border border-accent/[0.07]" />
            <div className="absolute inset-16 rounded-full border border-dashed border-accent/[0.12] motion-safe:animate-[spin_60s_linear_infinite]" />
          </div>
          <HeroGlobeClient />
        </motion.div>
      </div>
    </section>
  );
}
