"use client";

/**
 * Recharts requires a client component -- extracted from `app/models/
 * page.tsx` (a Server Component) after that page 500'd in dev with
 * "createContext is not a function": Recharts internals cannot run inside
 * React Server Component module evaluation. Every Recharts-using component
 * in this codebase must be marked "use client" for this reason.
 */
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { ModelPerformanceResponse } from "@/lib/api";

export default function ClassificationComparisonChart({
  models,
}: {
  models: ModelPerformanceResponse["classification"]["models"];
}) {
  const data = models.map((m) => ({
    name: m.display_name,
    macro_f1: m.metrics?.macro_f1 != null ? Number(m.metrics.macro_f1.toFixed(3)) : null,
    accuracy: m.metrics?.accuracy != null ? Number(m.metrics.accuracy.toFixed(3)) : null,
  }));
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
          <XAxis dataKey="name" stroke="rgba(255,255,255,0.25)" tick={{ fill: "#9BA6B8", fontSize: 11 }} />
          <YAxis stroke="rgba(255,255,255,0.25)" tick={{ fill: "#9BA6B8", fontSize: 11 }} width={40} />
          <Tooltip
            contentStyle={{ background: "#0B0F17", border: "1px solid rgba(255,255,255,0.09)", fontSize: 12 }}
            labelStyle={{ color: "#F2F5FA" }}
          />
          <Bar dataKey="macro_f1" name="Macro-F1" fill="#4C8DFF" isAnimationActive={false} />
          <Bar dataKey="accuracy" name="Accuracy" fill="#22D3A7" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
