# How AskPolicy Works

This document explains how AskPolicy works end to end: how documents are indexed, how questions are answered, and how the frontend, backend, vector database, and language model work together.

## Overview

AskPolicy is a Retrieval-Augmented Generation (RAG) application for company policy documents. Instead of asking a language model to answer from general knowledge, the system:

1. **Indexes** policy files into a searchable vector index (offline, via a CLI command).
2. **Retrieves** the most relevant document chunks for a user question.
3. **Generates** an answer using only those retrieved excerpts, with source references.

This keeps answers grounded in your documents and makes them verifiable.

## Main components

| Component | Role | Key implementation |
| --- | --- | --- |
| React frontend | Chat UI, document list/filter, session transcript, loading/error states | `frontend/src/App.tsx`, `hooks/useChat.ts`, `hooks/useWorkspace.ts`, `services/api.ts` |
| FastAPI backend | HTTP API, validation, orchestration, safe error responses | `backend/app/main.py`, `app/api/routes/*` |
| Ingestion CLI | Loads, chunks, embeds, and publishes document data to Qdrant | `backend/ingestion/ingest.py`, `loaders.py`, `chunker.py`, `metadata.py` |
| Embedding service | Local CPU embeddings for chunks and questions (`all-MiniLM-L6-v2`, 384 dims) | `backend/app/services/embedding_service.py`, `app/core/model_cache.py` |
| Qdrant Cloud | Stores vectors + payloads and performs similarity search | `backend/app/services/qdrant_service.py`, `app/services/retrieval_service.py`, ingestion |
| LLM service | Generates grounded answers via Groq (configurable model) | `backend/app/services/llm_service.py`, `app/core/prompts.py` |
| RAG + memory | Retrieval, prompt assembly, answer generation, short-term conversation memory | `backend/app/services/rag_service.py` (LangGraph `MessagesState` + `InMemorySaver`) |

Configuration is centralized in `backend/app/core/config.py` (loaded from root `.env`).

## 1. Document ingestion (how data gets into the system)

Ingestion is a separate, terminal-driven process. It indexes the current contents of `documents/` and makes them searchable through a Qdrant collection alias (default: `askpolicy_documents`).

**Command:** `cd backend && uv run python -m ingestion.ingest`

### Step-by-step flow

1. **Discover files** — Recursively scans `documents/`, ignores hidden paths and unsupported extensions, and processes `.pdf`, `.txt`, and `.docx` files (`ingestion/loaders.py`).
2. **Extract text**
   - **PDF:** one extracted unit per page (real page numbers preserved). Password-protected PDFs that cannot be opened are treated as errors.
   - **TXT:** one unit per file (decoded with common text encodings).
   - **DOCX:** one unit per file, including paragraphs and table rows in document order (no invented page numbers).
   Files with no extractable text are skipped and reported.
3. **Chunk text** — Uses LangChain `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`, separators `["\n\n", "\n", " "]` and fallback) to create overlapping, readable chunks (`ingestion/chunker.py`).
4. **Attach metadata** — Each chunk carries real metadata: `source` (relative path), `file_name`, `file_type`, `chunk_index`, and `page_number` only when available (PDF). No page numbers are invented for TXT/DOCX (`ingestion/metadata.py`).
5. **Generate embeddings locally** — The embedding model runs on CPU and is cached under `backend/.cache/huggingface/`. The same model is used later for query embeddings. Embedding dimension (384) is verified before indexing.
6. **Build and validate a new index safely**
   - Creates a temporary physical Qdrant collection with cosine distance.
   - Uploads points in batches (vector + `page_content` + `metadata` payload).
   - Validates point count and runs a probe query.
7. **Publish atomically** — Moves the configured alias to the new collection. The backend queries the alias, so updated data is available without restarting the backend.
8. **Clean up** — Deletes the previous physical collection and orphaned temporary ingestion collections after a successful switch.

### Safety behavior

- **Full re-index of current folder:** each successful run represents the current contents of `documents/`. To remove a document from search, remove it from `documents/` and re-run ingestion.
- **Full removal:** `uv run python -m ingestion.delete_index` deletes the alias, its collection, and any leftover temporary collections when you intentionally want no indexed data in Qdrant. It does not delete files in `documents/`.
- **No destructive pre-delete:** if loading, embedding, upload, validation, or activation fails, the previous alias/index remains active.
- **No valid documents:** if nothing supported/extractable is found, ingestion does not erase the existing index.

## 2. Question answering (how a chat response is produced)

When a user asks a question, the frontend calls `POST /api/chat` (`backend/app/api/routes/chat.py`). The RAG pipeline runs in `RagService` (`backend/app/services/rag_service.py`).

### Step-by-step flow

1. **Validate request** — Pydantic schemas enforce a non-empty question and validate optional fields (`history`, `source`, `thread_id`).
2. **Resolve conversation thread** — Uses the provided `thread_id` or generates one. If a new/restarted thread has no server state but client `history` is provided, the last exchanges are seeded once (server-side memory is authoritative after that).
3. **Embed the question** — The question is converted to a 384-dimensional vector using the same local embedding model as ingestion.
4. **Retrieve relevant chunks** — `RetrievalService` performs dense cosine similarity search in Qdrant via the collection alias (`top_k` configurable, default 5). If a document `source` filter is provided, Qdrant filters by `metadata.source` before applying the top-k limit.
5. **Handle "no evidence" early** — If retrieval returns no chunks, the backend returns the exact not-found message and **does not call the LLM** (no sources returned).
6. **Build a grounded prompt** — `app/core/prompts.py` constructs:
   - a **system prompt** with strict rules (handle greetings and unrelated messages conversationally, answer policy questions only from excerpts, no guessing/invention, cite sources, treat document text as untrusted, prefer the latest information when excerpts conflict, refuse when insufficient);
   - a **user prompt** containing recent conversation context (labeled as unverified), numbered retrieved excerpts with metadata, and the question.
7. **Generate the answer** — `LLMService` calls the configured Groq chat model and returns normalized text. If the provider fails or returns empty output, the API returns a safe error.
8. **Select sources for the response**
   - If the model returns the required not-found message, sources are cleared.
   - If the answer includes citation markers (e.g. `[1]`, `[2]`), only the cited retrieved chunks are returned.
   - Non-standard markers (e.g. `【3†L5-L6】`) are normalized to bracketed numbers (`[3]`) and any unmappable marker is stripped, so only `[n]` citations are shown.
   - If no valid citation markers are present, no sources are returned (this covers greetings and other non-policy replies).
9. **Update conversation memory and return** — The exchange is stored in thread memory (trimmed to the last 5 question/answer exchanges). The API returns `{ answer, sources, thread_id }`.

### Retrieval and grounding details

- **Dense semantic search:** questions and chunks are matched by vector similarity, not keyword-only lookup.
- **Metadata filtering:** users can scope questions to a specific document from the UI.
- **Source transparency:** returned sources come from actual retrieved chunks (file, path, page when available, chunk index, excerpt/score).
- **Refusal behavior:** unsupported questions produce: `I couldn't find this information in the available company documents.`
- **Greetings and small talk:** messages unrelated to policy receive a short conversational reply with no sources.

## 3. Conversation memory

- Implemented with **LangGraph** (`MessagesState` + `InMemorySaver`) in `RagService` as a single-node flow.
- Each chat uses a **`thread_id`**; memory is kept per thread.
- Retains at most the **last 5 question/answer exchanges**.
- **`POST /api/chat/reset`** clears server-side memory for a thread (used by “New chat” / clear actions).
- Memory is **in-process**: restarting the backend clears threads. The frontend can re-seed a restarted thread from session history.

Conversation history helps interpret follow-up questions; retrieval uses the current question text.

## 4. Frontend behavior

- **Workspace state** (`useWorkspace`): fetches health and indexed document metadata from the API, polls health periodically, and refreshes documents on demand.
- **Chat state** (`useChat`): manages messages, loading/error state, `thread_id`, and `sessionStorage` persistence for the current browser session; sends `question`, `history`, `source`, and `thread_id` to `/api/chat`.
- **UI**: sidebar (documents, counts, status, document filter, refresh/new chat), chat window with Markdown rendering, loading indicator, source references, copy/clear actions, and a responsive layout.
- **Theme** (`useTheme`): a light/dark theme toggle in the header and sidebar. The choice is applied as a `dark` class on the root element and persisted in `localStorage`, falling back to the operating system preference on first visit.
- **Data boundary:** the browser only talks to the FastAPI backend; Qdrant and Groq credentials remain server-side.

## 5. API and error handling

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Reports API/model info and Qdrant connectivity/readiness (lightweight check) |
| `GET /api/documents` | Returns indexed documents and chunk counts from Qdrant payloads |
| `POST /api/chat` | Returns grounded answer, selected sources, and `thread_id` |
| `POST /api/chat/reset` | Clears in-memory conversation state for a thread |

- Domain errors are mapped to meaningful HTTP responses (e.g. validation `422`, LLM failure `502`, unavailable/missing index or dependency issues `503`, unexpected `500`).
- Public responses avoid leaking internal exception details or credentials; details are logged server-side.

## 6. Configuration and local data

- **Environment:** root `.env` (see `.env.example`) provides Groq/Qdrant credentials, model names, chunk/retrieval settings, and app origins.
- **Local-only artifacts:** `backend/.venv`, `backend/.cache/huggingface`, `frontend/node_modules`, and `documents/` contents are excluded from version control.
- **Model cache:** embedding weights are stored in the project-managed cache (`backend/.cache/huggingface/`) rather than a global user cache.

## 7. End-to-end summary

1. Put `.pdf/.txt/.docx` files in `documents/`.
2. Run ingestion to chunk, embed locally, and publish vectors to Qdrant behind an alias.
3. Start backend and frontend.
4. User asks a question → backend retrieves relevant chunks (optionally filtered) → Groq generates a grounded answer → UI displays the answer and cited sources.
5. Re-run ingestion when documents change; the alias swap updates searchable data without a backend restart.

For setup commands, ingestion/reset procedures, and troubleshooting, see `docs/RUNNING_AND_DATA.md`. For deeper design rationale, see `docs/ARCHITECTURE.md`.