import {
  AlertCircle,
  Check,
  FileSearch,
  FileText,
  Files,
  Plus,
  RefreshCw,
  X,
} from 'lucide-react'

import type {
  BackendState,
  DocumentState,
  HealthResponse,
  IndexedDocument,
} from '../types'
import { formatDocumentSubtitle, truncateFileName } from '../utils/formatters'
import { Logo } from './Logo'

interface SidebarProps {
  open: boolean
  onClose: () => void
  documents: IndexedDocument[]
  documentState: DocumentState
  backendState: BackendState
  health: HealthResponse | null
  activeSource: string | null
  isRefreshing: boolean
  isBusy: boolean
  onSelectSource: (source: string | null) => void
  onNewChat: () => void
  onRefresh: () => void
}

const STATUS_COPY: Record<
  BackendState,
  { label: string; dot: string; tone: string }
> = {
  checking: {
    label: 'Checking connection',
    dot: 'bg-stone-400 animate-pulse',
    tone: 'text-ink-soft',
  },
  online: {
    label: 'Connected',
    dot: 'bg-emerald-500',
    tone: 'text-ink',
  },
  degraded: {
    label: 'Setup required',
    dot: 'bg-amber-500',
    tone: 'text-ink',
  },
  offline: {
    label: 'Backend offline',
    dot: 'bg-rose-500',
    tone: 'text-ink',
  },
}

export function Sidebar({
  open,
  onClose,
  documents,
  documentState,
  backendState,
  health,
  activeSource,
  isRefreshing,
  isBusy,
  onSelectSource,
  onNewChat,
  onRefresh,
}: SidebarProps) {
  const status = STATUS_COPY[backendState]
  const totalChunks = documents.reduce(
    (total, document) => total + document.chunk_count,
    0,
  )

  let statusDetail = 'Connecting to the API'
  if (backendState === 'online') {
    statusDetail = `${documents.length} ${documents.length === 1 ? 'document' : 'documents'} indexed`
  } else if (backendState === 'degraded') {
    statusDetail = health?.qdrant?.collection_ready
      ? 'Language model not configured'
      : 'No documents indexed yet'
  } else if (backendState === 'offline') {
    statusDetail = 'Start the FastAPI server'
  }

  return (
    <>
      {open && (
        <button
          type="button"
          aria-label="Close navigation"
          onClick={onClose}
          className="fixed inset-0 z-30 bg-ink/20 backdrop-blur-[1px] lg:hidden"
        />
      )}

      <aside
        aria-label="Workspace navigation"
        className={`fixed inset-y-0 left-0 z-40 flex w-72 max-w-[86vw] flex-col border-r border-line bg-[#f6f4ef] transition-transform duration-200 ease-out lg:static lg:z-auto lg:w-72 lg:max-w-none lg:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex h-16 shrink-0 items-center gap-2.5 border-b border-line/80 px-5">
          <Logo />
          <div className="min-w-0">
            <p className="text-[15px] font-semibold tracking-[-0.01em] text-ink">
              AskPolicy
            </p>
            <p className="text-[11px] text-ink-soft">Company policy assistant</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close sidebar"
            className="ml-auto -mr-1.5 inline-flex size-8 items-center justify-center rounded-lg text-ink-soft transition hover:bg-black/5 hover:text-ink lg:hidden"
          >
            <X size={17} />
          </button>
        </div>

        <div className="px-4 pt-4">
          <button
            type="button"
            onClick={onNewChat}
            disabled={isBusy}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-line bg-surface px-3 py-2.5 text-[13px] font-medium text-ink shadow-[0_1px_2px_rgba(32,37,35,0.04)] transition hover:border-accent-line hover:bg-accent-soft/60 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Plus size={15} strokeWidth={2.2} />
            New chat
          </button>
        </div>

        <div className="mt-7 flex items-center justify-between px-5">
          <h2 className="text-[10.5px] font-semibold uppercase tracking-[0.12em] text-ink-soft">
            Indexed documents
          </h2>
          <div className="flex items-center gap-1.5">
            {documents.length > 0 && (
              <span className="rounded-full border border-line bg-surface px-1.5 py-0.5 text-[10px] font-medium tabular-nums text-ink-soft">
                {documents.length}
              </span>
            )}
            <button
              type="button"
              onClick={onRefresh}
              disabled={isRefreshing}
              aria-label="Refresh indexed documents"
              title="Refresh indexed documents"
              className="inline-flex size-6 items-center justify-center rounded-md text-ink-soft transition hover:bg-black/5 hover:text-ink disabled:opacity-40"
            >
              <RefreshCw
                size={12.5}
                className={isRefreshing ? 'animate-spin' : ''}
              />
            </button>
          </div>
        </div>

        <nav className="mt-2 min-h-0 flex-1 overflow-y-auto px-3 pb-4">
          <button
            type="button"
            onClick={() => onSelectSource(null)}
            className={`mb-1 flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left transition ${
              activeSource === null
                ? 'bg-accent-soft text-ink'
                : 'text-ink-soft hover:bg-black/[0.035] hover:text-ink'
            }`}
          >
            <Files size={15} className="shrink-0" />
            <span className="min-w-0 flex-1">
              <span className="block text-[12.5px] font-medium">
                All documents
              </span>
              <span className="block text-[10.5px] text-ink-soft">
                {totalChunks} {totalChunks === 1 ? 'chunk' : 'chunks'} searchable
              </span>
            </span>
            {activeSource === null && (
              <Check size={13} className="shrink-0 text-accent" />
            )}
          </button>

          {documentState === 'loading' && (
            <div className="space-y-2 px-1 pt-2" aria-label="Loading documents">
              {[0, 1, 2].map((item) => (
                <div key={item} className="animate-pulse">
                  <div className="h-3 w-3/4 rounded bg-stone-200/80" />
                  <div className="mt-1.5 h-2 w-1/3 rounded bg-stone-200/60" />
                </div>
              ))}
            </div>
          )}

          {documentState === 'empty' && (
            <div className="mx-1 mt-3 rounded-xl border border-dashed border-line bg-surface/50 px-3 py-4 text-center">
              <FileSearch size={18} className="mx-auto text-ink-soft/70" />
              <p className="mt-2 text-[12px] font-medium text-ink">
                No documents indexed
              </p>
              <p className="mt-1 text-[10.5px] leading-4 text-ink-soft">
                Add files to <code className="font-mono">documents/</code> and run
                the ingestion command.
              </p>
            </div>
          )}

          {documentState === 'error' && (
            <div className="mx-1 mt-3 rounded-xl border border-danger/20 bg-danger/5 px-3 py-4 text-center">
              <AlertCircle size={18} className="mx-auto text-danger" />
              <p className="mt-2 text-[12px] font-medium text-ink">
                Could not load documents
              </p>
              <button
                type="button"
                onClick={onRefresh}
                className="mt-2 text-[11px] font-medium text-accent underline decoration-accent/30 underline-offset-2 hover:decoration-accent"
              >
                Try again
              </button>
            </div>
          )}

          {documents.map((document) => {
            const active = activeSource === document.source
            return (
              <button
                key={document.source}
                type="button"
                onClick={() => onSelectSource(active ? null : document.source)}
                title={document.source}
                className={`flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left transition ${
                  active
                    ? 'bg-accent-soft text-ink'
                    : 'text-ink-soft hover:bg-black/[0.035] hover:text-ink'
                }`}
              >
                <FileText size={15} className="shrink-0" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[12.5px] font-medium">
                    {truncateFileName(document.file_name)}
                  </span>
                  <span className="block text-[10.5px] text-ink-soft">
                    {formatDocumentSubtitle(document)}
                  </span>
                </span>
                {active && <Check size={13} className="shrink-0 text-accent" />}
              </button>
            )
          })}
        </nav>

        <div className="shrink-0 border-t border-line/80 px-5 py-4">
          <div className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className={`size-1.5 shrink-0 rounded-full ${status.dot}`}
            />
            <p className={`text-[12px] font-medium ${status.tone}`}>
              {status.label}
            </p>
          </div>
          <p className="mt-0.5 pl-[14px] text-[10.5px] text-ink-soft">
            {statusDetail}
          </p>
          <p className="mt-3 text-[10px] leading-4 text-ink-soft/80">
            Documents are managed from the terminal. AskPolicy searches only the
            indexed files.
          </p>
        </div>
      </aside>
    </>
  )
}
