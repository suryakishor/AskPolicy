import { Check, Copy } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'

import type { UiMessage } from '../types'
import { Logo } from './Logo'
import { SourceReferences } from './SourceReferences'

const MARKDOWN_COMPONENTS: Components = {
  a: ({ node: _node, ...props }) => (
    <a {...props} target="_blank" rel="noopener noreferrer" />
  ),
  table: ({ node: _node, children }) => (
    <div className="markdown-table-wrap">
      <table>{children}</table>
    </div>
  ),
}

interface ChatMessageProps {
  message: UiMessage
}

export function ChatMessage({ message }: ChatMessageProps) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-[14.5px] leading-6 text-on-accent shadow-[0_1px_2px_rgba(32,37,35,0.12)] sm:max-w-[75%]">
          {message.content}
        </div>
      </div>
    )
  }

  return <AssistantMessage message={message} />
}

function AssistantMessage({ message }: ChatMessageProps) {
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'error'>(
    'idle',
  )
  const timeoutRef = useRef<number | null>(null)

  useEffect(
    () => () => {
      if (timeoutRef.current) {
        window.clearTimeout(timeoutRef.current)
      }
    },
    [],
  )

  async function copyAnswer() {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopyState('copied')
    } catch {
      setCopyState('error')
    }
    if (timeoutRef.current) {
      window.clearTimeout(timeoutRef.current)
    }
    timeoutRef.current = window.setTimeout(() => setCopyState('idle'), 1800)
  }

  const copyLabel =
    copyState === 'copied'
      ? 'Copied'
      : copyState === 'error'
        ? 'Copy failed'
        : 'Copy answer'

  return (
    <div className="flex gap-3 sm:gap-3.5">
      <Logo size="sm" className="mt-0.5" />
      <div className="min-w-0 flex-1">
        <div className="mb-1.5 flex items-start justify-between gap-3">
          <span className="text-[11px] font-semibold tracking-wide text-ink-soft">
            AskPolicy
          </span>
          <button
            type="button"
            onClick={copyAnswer}
            title={copyLabel}
            aria-label={copyLabel}
            className="-mr-1 inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-[10px] font-medium text-ink-soft transition hover:bg-black/5 hover:text-ink dark:hover:bg-white/10"
          >
            {copyState === 'copied' ? (
              <Check size={12} className="text-accent" />
            ) : (
              <Copy size={12} />
            )}
            <span className="hidden sm:inline">{copyLabel}</span>
          </button>
        </div>

        <div className="markdown">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={MARKDOWN_COMPONENTS}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        <SourceReferences sources={message.sources} />
      </div>
    </div>
  )
}
