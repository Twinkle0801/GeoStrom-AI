export default function ErrorState({
  title, detail, onRetry,
}: {
  title: string;
  detail?: string;
  onRetry?: () => void;
}) {
  return (
    <div
      role="alert"
      className="flex flex-col items-start gap-2 rounded-lg border border-red-500/25 bg-red-500/[0.06] px-4 py-4 text-sm"
    >
      <p className="font-medium text-red-300">{title}</p>
      {detail && <p className="text-xs text-red-400/80">{detail}</p>}
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="rounded-md border border-red-500/30 px-3 py-1 text-xs font-medium text-red-300 transition-colors hover:bg-red-500/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-400"
        >
          Retry
        </button>
      )}
    </div>
  );
}
