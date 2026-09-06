import Badge from "@/components/ui/Badge";
import { CheckCircleIcon, WarningIcon } from "@/components/ui/Icons";
import type { ExplainResponse } from "@/lib/api";

export default function GroundingBadge({ result }: { result: ExplainResponse }) {
  if (result.source === "gemini") {
    return (
      <Badge tone="recommended">
        <CheckCircleIcon width={11} height={11} />
        Grounded in stored model output
      </Badge>
    );
  }
  return (
    <Badge tone="warning">
      <WarningIcon width={11} height={11} />
      Deterministic evidence summary
    </Badge>
  );
}
