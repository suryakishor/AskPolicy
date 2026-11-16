/**
 * Workspace metadata: backend health (polled) and the documents currently
 * indexed, read from Qdrant through the API.
 */

import { useCallback, useEffect, useState } from 'react'

import { ApiError, api } from '../services/api'
import type {
  BackendState,
  DocumentState,
  HealthResponse,
  IndexedDocument,
} from '../types'

const HEALTH_POLL_MS = 30_000

export function useWorkspace() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [backendState, setBackendState] = useState<BackendState>('checking')
  const [documents, setDocuments] = useState<IndexedDocument[]>([])
  const [documentState, setDocumentState] = useState<DocumentState>('loading')
  const [isRefreshing, setIsRefreshing] = useState(false)

  const refreshHealth = useCallback(async () => {
    try {
      const result = await api.health()
      setHealth(result)
      setBackendState(result.status === 'ok' ? 'online' : 'degraded')
    } catch {
      setHealth(null)
      setBackendState('offline')
    }
  }, [])

  const refreshDocuments = useCallback(async (showLoading = false) => {
    if (showLoading) {
      setDocumentState('loading')
    }
    try {
      const result = await api.listDocuments()
      setDocuments(result.documents)
      setDocumentState(result.documents.length > 0 ? 'ready' : 'empty')
    } catch (error) {
      setDocuments([])
      // 503 means ingestion has not run yet; anything else is a real failure.
      if (error instanceof ApiError && error.status === 503) {
        setDocumentState('empty')
      } else {
        setDocumentState('error')
      }
    }
  }, [])

  const refresh = useCallback(async () => {
    setIsRefreshing(true)
    try {
      await Promise.allSettled([refreshHealth(), refreshDocuments()])
    } finally {
      setIsRefreshing(false)
    }
  }, [refreshDocuments, refreshHealth])

  useEffect(() => {
    let active = true

    // Initial load: health decides the sidebar indicator, documents fill the list.
    void (async () => {
      await refreshHealth()
      if (active) {
        await refreshDocuments(true)
      }
    })()

    const interval = window.setInterval(() => {
      void refreshHealth()
    }, HEALTH_POLL_MS)

    return () => {
      active = false
      window.clearInterval(interval)
    }
  }, [refreshDocuments, refreshHealth])

  return {
    health,
    backendState,
    documents,
    documentState,
    isRefreshing,
    refresh,
  }
}
