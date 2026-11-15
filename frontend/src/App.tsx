import { Filter, Menu, Trash2, X } from 'lucide-react'
import { useState } from 'react'

import { ChatInput } from './components/ChatInput'
import { ChatWindow } from './components/ChatWindow'
import { Logo } from './components/Logo'
import { Sidebar } from './components/Sidebar'
import { useChat } from './hooks/useChat'
import { useWorkspace } from './hooks/useWorkspace'

export default function App() {
  const chat = useChat()
  const workspace = useWorkspace()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [selectedSource, setSelectedSource] = useState<string | null>(null)

  // Derive the active filter during render. If a re-ingestion removes the
  // selected file, the UI automatically falls back to "All documents" without
  // an effect or a stale filter request.
  const sourceFilter =
    selectedSource &&
    workspace.documents.some(
      (document) => document.source === selectedSource,
    )
      ? selectedSource
      : null
  const activeDocument =
    workspace.documents.find((document) => document.source === sourceFilter) ??
    null
  const hasDocuments = workspace.documents.length > 0
  const totalChunks = workspace.documents.reduce(
    (total, document) => total + document.chunk_count,
    0,
  )

  function handleNewChat() {
    setSelectedSource(null)
    setSidebarOpen(false)
    void chat.clearChat()
  }

  function handleSelectSource(source: string | null) {
    setSelectedSource(source)
    setSidebarOpen(false)
  }

  function handleSend(question: string) {
    void chat.sendMessage(question, sourceFilter)
  }

  const scopeLabel = hasDocuments
    ? `${workspace.documents.length} ${
        workspace.documents.length === 1 ? 'document' : 'documents'
      } · ${totalChunks} chunks`
    : 'No documents indexed'

  return (
    <div className="flex h-dvh overflow-hidden bg-canvas text-ink">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        documents={workspace.documents}
        documentState={workspace.documentState}
        backendState={workspace.backendState}
        health={workspace.health}
        activeSource={sourceFilter}
        isRefreshing={workspace.isRefreshing}
        isBusy={chat.isLoading}
        onSelectSource={handleSelectSource}
        onNewChat={handleNewChat}
        onRefresh={() => void workspace.refresh()}
      />

      <main className="flex min-h-0 min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center gap-2 border-b border-line bg-canvas/90 px-3 backdrop-blur sm:px-4">
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open navigation"
            className="-ml-1 inline-flex size-9 items-center justify-center rounded-lg text-ink-soft transition hover:bg-black/5 hover:text-ink lg:hidden"
          >
            <Menu size={18} />
          </button>

          <div className="hidden min-w-0 lg:block">
            <div className="flex items-baseline gap-2">
              <h1 className="text-[13px] font-semibold tracking-[-0.01em]">
                Policy assistant
              </h1>
              <span className="truncate text-[10.5px] text-ink-soft">
                {scopeLabel}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 lg:hidden">
            <Logo size="sm" />
            <span className="text-[13px] font-semibold tracking-[-0.01em]">
              AskPolicy
            </span>
          </div>

          <div className="ml-auto flex min-w-0 items-center gap-2">
            {activeDocument && (
              <div className="flex min-w-0 max-w-[52vw] items-center gap-1.5 rounded-lg border border-accent-line bg-accent-soft px-2 py-1.5 text-[10.5px] font-medium text-accent sm:max-w-[280px]">
                <Filter size={11} className="shrink-0" />
                <span className="truncate" title={activeDocument.source}>
                  {activeDocument.file_name}
                </span>
                <button
                  type="button"
                  onClick={() => setSelectedSource(null)}
                  aria-label="Clear document filter"
                  className="-mr-0.5 inline-flex size-4 shrink-0 items-center justify-center rounded text-accent/70 transition hover:bg-accent/10 hover:text-accent"
                >
                  <X size={11} />
                </button>
              </div>
            )}

            <button
              type="button"
              onClick={() => void chat.clearChat()}
              disabled={chat.messages.length === 0 || chat.isLoading}
              title="Clear conversation"
              className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 text-[11px] font-medium text-ink-soft transition hover:border-accent-line hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Trash2 size={13} />
              <span className="hidden sm:inline">Clear</span>
            </button>
          </div>
        </header>

        <ChatWindow
          messages={chat.messages}
          isLoading={chat.isLoading}
          hasDocuments={hasDocuments}
          backendState={workspace.backendState}
          onSend={handleSend}
        />

        <ChatInput
          onSend={handleSend}
          disabled={chat.isLoading}
          isLoading={chat.isLoading}
          error={chat.error}
          onDismissError={chat.dismissError}
        />
      </main>
    </div>
  )
}
