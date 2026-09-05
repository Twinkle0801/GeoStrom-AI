import Badge from "@/components/ui/Badge";
import type { ExplainResponse } from "@/lib/api";

export default function GroundingBadge({ result }: { result: ExplainResponse }) {
  if (result.source === "gemini") {
    return <Badge tone="recommended">Grounded in stored model output</Badge>;
  }
  return <Badge tone="warning">Deterministic evidence summary</Badge>;
}
