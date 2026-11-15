import { ChevronDown, FileText } from 'lucide-react'

import type { SourceReference } from '../types'
import { formatSourceLocation } from '../utils/formatters'

interface SourceReferencesProps {
  sources: SourceReference[]
}

/**
 * Evidence shown beneath an assistant answer.
 *
 * Every entry comes directly from the backend's retrieved chunks. Nothing here
 * is generated in the browser, so citations always map to real indexed text.
 */
export function SourceReferences({ sources }: SourceReferencesProps) {
  if (sources.length === 0) {
    return null
  }

  const documentCount = new Set(sources.map((source) => source.source)).size

  return (
    <section aria-label="Source references" className="mt-4">
      <div className="mb-2 flex items-center gap-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-soft">
          Sources
        </span>
        <span className="rounded-full border border-line bg-surface px-1.5 py-px text-[9.5px] font-medium tabular-nums text-ink-soft">
          {documentCount} {documentCount === 1 ? 'document' : 'documents'}
        </span>
      </div>

      <div className="space-y-1.5">
        {sources.map((source, index) => (
          <details
            key={`${source.source}-${source.page_number ?? 'na'}-${source.chunk_index}-${index}`}
            className="group overflow-hidden rounded-lg border border-line bg-surface"
          >
            <summary className="flex cursor-pointer list-none items-center gap-2.5 px-3 py-2.5 marker:hidden hover:bg-black/[0.02] [&::-webkit-details-marker]:hidden">
              <FileText size={14} className="shrink-0 text-accent" />
              <span className="min-w-0 flex-1">
                <span
                  className="block truncate text-[11.5px] font-medium text-ink"
                  title={source.source}
                >
                  {source.file_name}
                </span>
                <span className="mt-0.5 block text-[10px] text-ink-soft">
                  {formatSourceLocation(
                    source.page_number,
                    source.chunk_index,
                  )}
                </span>
              </span>
              <ChevronDown
                size={13}
                className="shrink-0 text-ink-soft transition-transform duration-150 group-open:rotate-180"
              />
              <span className="sr-only">Toggle source excerpt</span>
            </summary>
            <div className="max-h-56 overflow-y-auto border-t border-line/80 bg-[#fbfaf7] px-3.5 py-3">
              <p className="whitespace-pre-wrap text-[11.5px] leading-[1.7] text-ink-soft">
                {source.content}
              </p>
            </div>
          </details>
        ))}
      </div>
    </section>
  )
}
