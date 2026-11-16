/** Shared frontend types. These mirror the FastAPI Pydantic schemas. */

export type MessageRole = 'user' | 'assistant'

export interface SourceReference {
  file_name: string
  source: string
  file_type: string
  page_number: number | null
  chunk_index: number
  content: string
  score: number
}

export interface ChatResponse {
  answer: string
  sources: SourceReference[]
  thread_id: string
}

export interface ChatMessage {
  role: MessageRole
  content: string
}

export interface ChatRequest {
  question: string
  history: ChatMessage[]
  source: string | null
  thread_id: string | null
}

export interface ResetResponse {
  cleared: boolean
  thread_id: string
}

export interface UiMessage {
  id: string
  role: MessageRole
  content: string
  sources: SourceReference[]
}

export interface IndexedDocument {
  file_name: string
  file_type: string
  source: string
  chunk_count: number
}

export interface DocumentListResponse {
  documents: IndexedDocument[]
  total_documents: number
  total_chunks: number
  collection: string
}

export interface QdrantHealth {
  configured?: boolean
  connected?: boolean
  collection_ready?: boolean
  collection?: string
  detail?: string
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  app_env: string
  embedding_model: string
  llm_model: string
  qdrant: QdrantHealth
  documents_indexed: boolean
}

export type BackendState = 'checking' | 'online' | 'degraded' | 'offline'
export type DocumentState = 'loading' | 'ready' | 'empty' | 'error'
