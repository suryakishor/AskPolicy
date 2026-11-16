import { AlertCircle, MessageSquareText, ShieldCheck } from 'lucide-react'

import type { BackendState } from '../types'
import { Logo } from './Logo'

const SUGGESTED_QUESTIONS = [
  "What is the company's leave policy?",
  'How does the reimbursement process work?',
  'What is the employee onboarding procedure?',
  "What are the company's travel guidelines?",
] as const

interface WelcomeScreenProps {
  onSelectQuestion: (question: string) => void
  disabled: boolean
  hasDocuments: boolean
  backendState: BackendState
}

export function WelcomeScreen({
  onSelectQuestion,
  disabled,
  hasDocuments,
  backendState,
}: WelcomeScreenProps) {
  const showOffline = backendState === 'offline'
  const showEmpty = !showOffline && backendState !== 'checking' && !hasDocuments

  return (
    <div className="mx-auto flex min-h-full w-full max-w-2xl flex-col items-center justify-center px-1 py-10 text-center sm:py-16">
      <Logo size="lg" />
      <h1 className="mt-5 font-serif text-[28px] leading-tight tracking-[-0.02em] text-ink sm:text-[34px]">
        How can I help you today?
      </h1>
      <p className="mt-2.5 max-w-md text-[13.5px] leading-6 text-ink-soft">
        Ask questions about your company&apos;s policies, procedures, and
        documentation. Answers are grounded in the documents indexed in your
        workspace.
      </p>

      {(showOffline || showEmpty) && (
        <div className="mt-5 flex w-full max-w-md items-start gap-2 rounded-xl border border-amber-600/20 bg-amber-50 px-3.5 py-2.5 text-left dark:border-amber-400/20 dark:bg-amber-500/10">
          <AlertCircle size={14} className="mt-0.5 shrink-0 text-amber-700 dark:text-amber-300" />
          <p className="text-[11px] leading-[1.6] text-amber-900 dark:text-amber-100">
            {showOffline ? (
              <>
                The backend is not reachable. Start it with{' '}
                <code className="rounded border border-amber-600/15 bg-amber-100/70 px-1 py-0.5 font-mono text-[10px] dark:border-amber-400/20 dark:bg-amber-500/15">
                  uv run uvicorn app.main:app --reload
                </code>
                .
              </>
            ) : (
              <>
                No documents are indexed yet. Add PDF, DOCX, or TXT files to{' '}
                <code className="rounded border border-amber-600/15 bg-amber-100/70 px-1 py-0.5 font-mono text-[10px] dark:border-amber-400/20 dark:bg-amber-500/15">
                  documents/
                </code>{' '}
                and run the ingestion command.
              </>
            )}
          </p>
        </div>
      )}

      <div className="mt-8 grid w-full grid-cols-1 gap-2.5 sm:grid-cols-2">
        {SUGGESTED_QUESTIONS.map((question) => (
          <button
            key={question}
            type="button"
            disabled={disabled}
            onClick={() => onSelectQuestion(question)}
            className="group flex items-start gap-2.5 rounded-xl border border-line bg-surface px-3.5 py-3 text-left shadow-[0_1px_2px_rgba(32,37,35,0.03)] transition hover:border-accent-line hover:bg-accent-soft/40 focus-visible:border-accent disabled:cursor-not-allowed disabled:opacity-50"
          >
            <MessageSquareText
              size={14}
              className="mt-0.5 shrink-0 text-ink-soft transition group-hover:text-accent"
            />
            <span className="text-[12.5px] font-medium leading-5 text-ink">
              {question}
            </span>
          </button>
        ))}
      </div>

      <p className="mt-8 flex items-center gap-1.5 text-[10.5px] text-ink-soft/80">
        <ShieldCheck size={12} className="shrink-0" />
        AskPolicy answers only from indexed documents. Verify important policy
        details.
      </p>
    </div>
  )
}
