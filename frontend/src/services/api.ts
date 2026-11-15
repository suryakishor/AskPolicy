/**
 * Thin typed wrapper around the AskPolicy FastAPI endpoints.
 *
 * The backend URL is configurable with VITE_API_BASE_URL. Only the API base
 * URL is exposed to the browser; Groq and Qdrant credentials stay server-side.
 */

import type {
  ChatRequest,
  ChatResponse,
  DocumentListResponse,
  HealthResponse,
  ResetResponse,
} from '../types'

const API_BASE_URL = (
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, '') ||
  'http://127.0.0.1:8000'
)

const HEALTH_TIMEOUT_MS = 8_000
const REQUEST_TIMEOUT_MS = 20_000
const CHAT_TIMEOUT_MS = 120_000

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

interface ValidationIssue {
  msg?: string
}

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as {
      detail?: string | ValidationIssue[]
    }
    if (typeof body.detail === 'string') {
      return body.detail
    }
    if (Array.isArray(body.detail)) {
      const messages = body.detail
        .map((issue) => issue.msg)
        .filter((msg): msg is string => Boolean(msg))
      if (messages.length > 0) {
        return messages.join('. ')
      }
    }
  } catch {
    // The response was not JSON (for example, a proxy error page).
  }

  if (response.status === 422) {
    return 'Please check the question and try again.'
  }
  return `The request failed (HTTP ${response.status}). Please try again.`
}

async function request<T>(
  path: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init.headers ?? {}),
      },
      signal: AbortSignal.timeout(timeoutMs),
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'TimeoutError') {
      throw new ApiError('The request took too long. Please try again.', 408)
    }
    throw new ApiError(
      'Could not reach the AskPolicy backend. Make sure it is running and try again.',
      0,
    )
  }

  if (!response.ok) {
    throw new ApiError(await readErrorMessage(response), response.status)
  }
  return (await response.json()) as T
}

export const api = {
  health(): Promise<HealthResponse> {
    return request<HealthResponse>(
      '/api/health',
      { method: 'GET' },
      HEALTH_TIMEOUT_MS,
    )
  },

  listDocuments(): Promise<DocumentListResponse> {
    return request<DocumentListResponse>(
      '/api/documents',
      { method: 'GET' },
      REQUEST_TIMEOUT_MS,
    )
  },

  ask(payload: ChatRequest): Promise<ChatResponse> {
    return request<ChatResponse>(
      '/api/chat',
      { method: 'POST', body: JSON.stringify(payload) },
      CHAT_TIMEOUT_MS,
    )
  },

  resetThread(threadId: string): Promise<ResetResponse> {
    return request<ResetResponse>(
      '/api/chat/reset',
      {
        method: 'POST',
        body: JSON.stringify({ thread_id: threadId }),
      },
      REQUEST_TIMEOUT_MS,
    )
  },
}
