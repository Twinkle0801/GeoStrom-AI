/**
 * A condensed restatement of methodology/page.tsx's own "why storm-level
 * splitting matters" / "retrospective, not operational" sections -- same
 * facts, shorter form, for the home page. The full explanation lives at
 * /methodology; this never contradicts or duplicates it verbatim.
 * Zero-JS server component -- see FadeIn.tsx.
 */
import Link from "next/link";
import GlassPanel from "@/components/ui/GlassPanel";
import SectionHeader from "@/components/ui/SectionHeader";
import FadeIn from "@/components/ui/FadeIn";
import { AlertTriangleIcon, ArrowRightIcon, CheckCircleIcon, DatabaseIcon } from "@/components/ui/Icons";

const CALLOUTS = [
  {
    icon: DatabaseIcon,
    title: "Storm-level split, frozen once",
    body: "Train/validation/test are split by whole storm, never by observation -- a storm's early life can never leak into a split containing its later life.",
  },
  {
    icon: CheckCircleIcon,
    title: "Evaluated once, on held-out storms",
    body: "Every model is scored exactly once against storms it never saw during training -- the same frozen split every time, for every model.",
  },
  {
    icon: AlertTriangleIcon,
    title: "Retrospective, not operational",
    body: "Every storm here has already happened. Every prediction is historical baseline output, evaluated after the fact -- not a live forecast, warning, or advisory.",
  },
] as const;

export default function ScientificIntegrity() {
  return (
    <section className="mx-auto max-w-7xl px-6 py-16">
      <SectionHeader
        eyebrow="Scientific methodology"
        title="Built to be checked, not just trusted"
        description="Three principles that hold across every page in this product."
      />
      <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
        {CALLOUTS.map((c, i) => (
          <FadeIn key={c.title} delayMs={i * 60}>
            <GlassPanel className="flex h-full flex-col gap-2 p-5">
              <c.icon width={18} height={18} className="text-accent-soft" />
              <h3 className="text-sm font-semibold text-text-primary">{c.title}</h3>
              <p className="text-sm leading-relaxed text-text-secondary">{c.body}</p>
            </GlassPanel>
          </FadeIn>
        ))}
      </div>
      <FadeIn className="mt-6">
        <Link
          href="/methodology"
          className="inline-flex items-center gap-1.5 text-sm font-medium text-accent-soft hover:text-accent"
        >
          Read the full methodology <ArrowRightIcon width={14} height={14} />
        </Link>
      </FadeIn>
    </section>
  );
}
