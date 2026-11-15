/**
 * Chat state for the current browser session.
 *
 * Responsibilities:
 * - keep messages in React state (and sessionStorage for a refresh-safe session);
 * - call POST /api/chat with the active thread id and optional document filter;
 * - surface loading and error states;
 * - clear both UI state and the backend's in-memory conversation.
 */

import { useCallback, useEffect, useState } from 'react'

import { ApiError, api } from '../services/api'
import type { ChatMessage, UiMessage } from '../types'

const STORAGE_KEY = 'askpolicy.session.v1'

interface PersistedSession {
  messages: UiMessage[]
  threadId: string | null
}

function createId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function loadSession(): PersistedSession {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return { messages: [], threadId: null }
    }
    const parsed = JSON.parse(raw) as Partial<PersistedSession>
    const messages = Array.isArray(parsed.messages)
      ? parsed.messages.filter(
          (message): message is UiMessage =>
            Boolean(message) &&
            (message.role === 'user' || message.role === 'assistant') &&
            typeof message.content === 'string' &&
            Array.isArray(message.sources),
        )
      : []
    return {
      messages,
      threadId: typeof parsed.threadId === 'string' ? parsed.threadId : null,
    }
  } catch {
    return { messages: [], threadId: null }
  }
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message
  }
  return 'Something went wrong while contacting the assistant. Please try again.'
}

export function useChat() {
  const [initialSession] = useState(loadSession)
  const [messages, setMessages] = useState<UiMessage[]>(initialSession.messages)
  const [threadId, setThreadId] = useState<string | null>(
    initialSession.threadId,
  )
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Persist only chat data; no credentials or environment values are stored.
  useEffect(() => {
    try {
      if (messages.length === 0 && !threadId) {
        sessionStorage.removeItem(STORAGE_KEY)
      } else {
        sessionStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({ messages, threadId }),
        )
      }
    } catch {
      // sessionStorage can be unavailable (private mode/quota); chat still works.
    }
  }, [messages, threadId])

  const sendMessage = useCallback(
    async (question: string, source: string | null) => {
      const trimmed = question.trim()
      if (!trimmed || isLoading) {
        return
      }

      setError(null)
      const userMessage: UiMessage = {
        id: createId(),
        role: 'user',
        content: trimmed,
        sources: [],
      }

      // Snapshot the prior turns before appending the new question. The backend
      // ignores this when its LangGraph thread already has state, but it lets a
      // restored browser session seed memory after a backend restart.
      const history: ChatMessage[] = messages.slice(-10).map((message) => ({
        role: message.role,
        content: message.content,
      }))

      setMessages((previous) => [...previous, userMessage])
      setIsLoading(true)

      try {
        const response = await api.ask({
          question: trimmed,
          history,
          source,
          thread_id: threadId,
        })
        const assistantMessage: UiMessage = {
          id: createId(),
          role: 'assistant',
          content: response.answer,
          sources: response.sources ?? [],
        }
        setMessages((previous) => [...previous, assistantMessage])
        setThreadId(response.thread_id)
      } catch (requestError) {
        setError(errorMessage(requestError))
      } finally {
        setIsLoading(false)
      }
    },
    [isLoading, messages, threadId],
  )

  const clearChat = useCallback(async () => {
    const previousThread = threadId
    setMessages([])
    setThreadId(null)
    setError(null)

    // Best-effort server cleanup. A new thread id is generated on the next
    // question, so a failed reset only leaves unused memory in the process.
    if (previousThread) {
      try {
        await api.resetThread(previousThread)
      } catch {
        // Ignore: local state is already cleared.
      }
    }
  }, [threadId])

  return {
    messages,
    threadId,
    isLoading,
    error,
    sendMessage,
    clearChat,
    dismissError: () => setError(null),
  }
}
