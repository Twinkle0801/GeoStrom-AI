export interface SatelliteFrameMeta {
  timestamp: string;
  sid: string;
  source: string;
  channel: string;
  qualityStatus: string;
  sceneLabel: string | null;
}

export default function SatelliteMetadata({ meta }: { meta: SatelliteFrameMeta }) {
  const rows: Array<[string, string]> = [
    ["Timestamp", meta.timestamp],
    ["Storm ID", meta.sid],
    ["Source", meta.source],
    ["Channel", meta.channel],
    ["Quality", meta.qualityStatus],
    ["Dvorak scene", meta.sceneLabel ?? "Not available"],
  ];
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-text-muted">{k}</dt>
          <dd className="text-right text-text-secondary">{v}</dd>
        </div>
      ))}
    </dl>
  );
}
