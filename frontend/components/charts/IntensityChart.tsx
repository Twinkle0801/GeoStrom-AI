"use client";

/**
 * Observed vs. predicted wind, for one selected forecast origin + model.
 * Two DISTINCT Recharts series (never merged into one line), per the task's
 * explicit rule -- observed is the real IBTrACS wind history (teal, solid,
 * always the full storm lifetime for context); predicted is this forecast's
 * +6/+12/+18/+24h points (amber, dashed connector, diamond markers), each
 * carrying its own tooltip identifying series type, units, and horizon.
 */
import {
  Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip,
  XAxis, YAxis,
} from "recharts";
import type { ObservationList, PredictionList } from "@/lib/api";
import { formatTimestamp } from "@/lib/format";

interface ChartPoint {
  tMs: number;
  ts: string;
  windKt: number;
  leadHours?: number;
}

export default function IntensityChart({
  observations, predictions, originTs,
}: {
  observations: ObservationList;
  predictions: PredictionList;
  originTs: string | null;
}) {
  const observed: ChartPoint[] = observations
    .filter((o) => o.wind_kt != null)
    .map((o) => ({ tMs: new Date(o.ts).getTime(), ts: o.ts, windKt: o.wind_kt as number }));

  const intensityPreds = predictions
    .filter((p) => p.task === "intensity" && p.pred_wind_kt != null)
    .sort((a, b) => a.lead_hours - b.lead_hours);

  const predicted: ChartPoint[] = intensityPreds.map((p) => ({
    tMs: new Date(p.valid_ts).getTime(), ts: p.valid_ts,
    windKt: p.pred_wind_kt as number, leadHours: p.lead_hours,
  }));

  if (observed.length === 0) {
    return null;
  }

  const originMs = originTs ? new Date(originTs).getTime() : null;

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart margin={{ top: 8, right: 16, bottom: 0, left: -12 }}>
          <defs>
            <linearGradient id="observedFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#22D3A7" stopOpacity={0.22} />
              <stop offset="95%" stopColor="#22D3A7" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
          <XAxis
            dataKey="tMs"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v: number) => new Date(v).toLocaleDateString("en-US", { month: "short", day: "2-digit" })}
            stroke="rgba(255,255,255,0.25)"
            tick={{ fill: "#9BA6B8", fontSize: 11 }}
          />
          <YAxis
            unit=" kt"
            stroke="rgba(255,255,255,0.25)"
            tick={{ fill: "#9BA6B8", fontSize: 11 }}
            width={52}
          />
          <Tooltip content={<IntensityTooltip />} cursor={{ stroke: "rgba(255,255,255,0.15)" }} />
          {originMs != null && (
            <ReferenceLine x={originMs} stroke="#7FB0FF" strokeDasharray="3 3" label={{
              value: "Forecast origin", position: "insideTopRight", fill: "#7FB0FF", fontSize: 10,
            }} />
          )}
          <Area
            data={observed} dataKey="windKt" stroke="none" fill="url(#observedFill)"
            isAnimationActive={false}
          />
          <Line
            data={observed} dataKey="windKt" name="Observed"
            stroke="#22D3A7" strokeWidth={2} dot={false} isAnimationActive={false}
          />
          <Line
            data={predicted} dataKey="windKt" name="Predicted"
            stroke="#FFB020" strokeWidth={2} strokeDasharray="6 6" dot={false}
            isAnimationActive={false}
          />
          <Scatter data={predicted} dataKey="windKt" name="Predicted" fill="#FFB020" shape="diamond" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function IntensityTooltip({
  active, payload,
}: {
  active?: boolean;
  payload?: Array<{ name?: string; value?: number; payload: ChartPoint }>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0];
  const isObserved = point.name === "Observed";
  const seriesType = isObserved ? "Observed (IBTrACS)" : "Predicted (model output)";
  return (
    <div className="rounded-md border border-border-subtle bg-bg-elevated px-3 py-2 text-xs shadow-elevated">
      <div className="font-medium text-text-primary">{formatTimestamp(point.payload.ts)}</div>
      <div className={`mt-0.5 font-medium ${isObserved ? "text-truth" : "text-predicted"}`}>{seriesType}</div>
      <div className="font-mono tabular-nums text-text-primary">{point.payload.windKt.toFixed(1)} kt</div>
      {point.payload.leadHours != null && (
        <div className="font-mono text-text-muted">+{point.payload.leadHours}h horizon</div>
      )}
    </div>
  );
}
