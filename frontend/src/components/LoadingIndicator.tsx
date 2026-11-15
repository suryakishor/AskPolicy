import { Logo } from './Logo'

/** Restrained typing indicator shown while the RAG request is in flight. */
export function LoadingIndicator() {
  return (
    <div className="flex items-start gap-3 sm:gap-3.5" aria-live="polite">
      <Logo size="sm" className="mt-0.5" />
      <div className="flex items-center gap-2.5 pt-1.5">
        <span className="flex items-center gap-1" aria-hidden="true">
          {[0, 1, 2].map((dot) => (
            <span
              key={dot}
              className="size-1.5 animate-pulse rounded-full bg-ink-soft/55"
              style={{ animationDelay: `${dot * 180}ms` }}
            />
          ))}
        </span>
        <span className="text-[12px] text-ink-soft">
          Searching company documents…
        </span>
      </div>
    </div>
  )
}
