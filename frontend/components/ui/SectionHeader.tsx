export default function SectionHeader({
  eyebrow, title, description,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
}) {
  return (
    <div>
      {eyebrow && (
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-accent-soft">
          <span aria-hidden className="h-px w-5 bg-accent-soft/50" />
          {eyebrow}
        </div>
      )}
      <h2 className="mt-2 text-xl font-semibold tracking-tight text-text-primary sm:text-2xl">
        {title}
      </h2>
      {description && (
        <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-text-secondary">{description}</p>
      )}
    </div>
  );
}
