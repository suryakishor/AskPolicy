import { AlertCircle, ArrowUp, Loader2, X } from 'lucide-react'
import {
  type FormEvent,
  type KeyboardEvent,
  useEffect,
  useRef,
  useState,
} from 'react'

interface ChatInputProps {
  onSend: (question: string) => void
  disabled: boolean
  isLoading: boolean
  error: string | null
  onDismissError: () => void
}

export function ChatInput({
  onSend,
  disabled,
  isLoading,
  error,
  onDismissError,
}: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Grow the textarea up to a comfortable limit, then scroll internally.
  useEffect(() => {
    const element = textareaRef.current
    if (!element) {
      return
    }
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 160)}px`
  }, [value])

  function submit() {
    const trimmed = value.trim()
    if (!trimmed || disabled) {
      return
    }
    onSend(trimmed)
    setValue('')
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    submit()
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter inserts a newline. isComposing prevents broken
    // behavior with IME input (for example, Japanese or Korean keyboards).
    if (
      event.key === 'Enter' &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault()
      submit()
    }
  }

  const canSend = !disabled && value.trim().length > 0

  return (
    <div className="shrink-0 border-t border-line bg-canvas/95 px-4 pb-4 pt-3 backdrop-blur sm:px-6 sm:pb-5">
      <div className="mx-auto w-full max-w-3xl">
        {error && (
          <div
            role="alert"
            className="mb-2.5 flex items-start gap-2 rounded-xl border border-danger/20 bg-danger/[0.055] px-3.5 py-2.5"
          >
            <AlertCircle size={14} className="mt-0.5 shrink-0 text-danger" />
            <p className="flex-1 text-[11.5px] leading-5 text-danger">{error}</p>
            <button
              type="button"
              onClick={onDismissError}
              aria-label="Dismiss error"
              className="-mr-1 inline-flex size-6 shrink-0 items-center justify-center rounded-md text-danger/70 transition hover:bg-danger/10 hover:text-danger dark:hover:bg-danger/20"
            >
              <X size={13} />
            </button>
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="flex items-end gap-2 rounded-2xl border border-line bg-surface p-2 shadow-[0_1px_2px_rgba(32,37,35,0.04),0_12px_32px_-24px_rgba(32,37,35,0.35)] transition focus-within:border-accent/35 focus-within:ring-4 focus-within:ring-accent/[0.06]">
            <textarea
              ref={textareaRef}
              value={value}
              onChange={(event) => setValue(event.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              aria-label="Ask a policy question"
              aria-describedby="chat-input-hint"
              placeholder="Ask about leave, reimbursement, travel, or onboarding…"
              className="max-h-40 min-h-[38px] flex-1 resize-none bg-transparent px-2.5 py-2 text-[14px] leading-6 text-ink outline-none placeholder:text-ink-soft/65"
            />
            <button
              type="submit"
              disabled={!canSend}
              aria-label={isLoading ? 'Sending question' : 'Send question'}
              className="mb-0.5 inline-flex size-9 shrink-0 items-center justify-center rounded-xl bg-accent text-white shadow-[0_1px_2px_rgba(28,91,79,0.25)] transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:bg-stone-200 disabled:text-stone-400 disabled:shadow-none"
            >
              {isLoading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <ArrowUp size={17} strokeWidth={2.2} />
              )}
            </button>
          </div>
        </form>

        <div
          id="chat-input-hint"
          className="mt-2 flex flex-wrap items-center justify-between gap-1 px-1 text-[10px] text-ink-soft/80"
        >
          <span>
            <kbd className="font-sans font-medium">Enter</kbd> to send ·{' '}
            <kbd className="font-sans font-medium">Shift + Enter</kbd> for a new
            line
          </span>
          <span className="hidden sm:inline">
            Answers are limited to indexed documents
          </span>
        </div>
      </div>
    </div>
  )
}
