import { useEffect, useRef } from 'react'

import type { BackendState, UiMessage } from '../types'
import { ChatMessage } from './ChatMessage'
import { LoadingIndicator } from './LoadingIndicator'
import { WelcomeScreen } from './WelcomeScreen'

interface ChatWindowProps {
  messages: UiMessage[]
  isLoading: boolean
  hasDocuments: boolean
  backendState: BackendState
  onSend: (question: string) => void
}

export function ChatWindow({
  messages,
  isLoading,
  hasDocuments,
  backendState,
  onSend,
}: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  // Keep the newest message in view as answers arrive.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    })
  }, [messages.length, isLoading])

  if (messages.length === 0 && !isLoading) {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto px-4 sm:px-6">
        <WelcomeScreen
          onSelectQuestion={onSend}
          disabled={isLoading}
          hasDocuments={hasDocuments}
          backendState={backendState}
        />
      </div>
    )
  }

  return (
    <div
      className="min-h-0 flex-1 overflow-y-auto px-4 sm:px-6"
      role="log"
      aria-live="polite"
      aria-relevant="additions"
      aria-label="Conversation"
    >
      <div className="mx-auto w-full max-w-3xl space-y-6 py-6 sm:py-8">
        {messages.map((message) => (
          <div key={message.id} className="message-enter">
            <ChatMessage message={message} />
          </div>
        ))}
        {isLoading && <LoadingIndicator />}
        <div ref={bottomRef} className="h-px" />
      </div>
    </div>
  )
}
