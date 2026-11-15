# AskPolicy: Run, Insert, and Delete Data

This guide explains how to run AskPolicy locally, how to add/index documents in Qdrant, and how to remove previously indexed data.

## 1. Prerequisites

- **uv** (Python environment/dependency manager)
- **Node.js 20+** and **npm**
- A **Qdrant Cloud** cluster (`QDRANT_URL`, `QDRANT_API_KEY`)
- A **Groq** API key (`GROQ_API_KEY`) and a supported chat model
- Internet access on first run to download the embedding model (`sentence-transformers/all-MiniLM-L6-v2`) into `backend/.cache/huggingface/`

## 2. One-time setup

### 2.1 Environment variables
1. Copy `.env.example` to `.env` in the repository root (`D:\AskPolicy\.env`).
2. Fill in your credentials (keep them server-side; never put them in frontend code):

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL="openai/gpt-oss-120b"

QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=askpolicy_documents

APP_ENV=development
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
FRONTEND_ORIGIN=http://localhost:5173

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5
```

> Do not overwrite an existing `.env` that already contains working credentials.

### 2.2 Install dependencies

**Backend (creates/uses `backend/.venv`)**
```powershell
cd D:\AskPolicy\backend
uv sync --dev
```

**Frontend (installs to `frontend/node_modules`)**
```powershell
cd D:\AskPolicy\frontend
npm install
```

## 3. How to run the project

Use separate terminals for the backend and frontend. Ingest documents first (or whenever documents change) so the chat has data to search.

### 3.1 Index documents (ingestion)
Place supported files in `D:\AskPolicy\documents\` (nested folders are supported). Supported formats: **`.pdf`, `.txt`, `.docx`**. Unsupported/hidden files are ignored; empty/unreadable files may be skipped or cause a safe failure depending on type.

```powershell
cd D:\AskPolicy\backend
uv run python -m ingestion.ingest
```

The command prints real statistics (discovered/processed/skipped, PDF pages, chunks, model, alias/collection status). You can re-run it any time the contents of `documents/` change **without restarting the backend**.

### 3.2 Start the backend API
```powershell
cd D:\AskPolicy\backend
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`
- Endpoints: `GET /api/health`, `GET /api/documents`, `POST /api/chat`, `POST /api/chat/reset`

### 3.3 Start the frontend
```powershell
cd D:\AskPolicy\frontend
npm run dev
```

- UI: `http://localhost:5173`
- The frontend calls the backend at `http://127.0.0.1:8000` by default. To override, set `VITE_API_BASE_URL` in `frontend/.env.local`.

### 3.4 Quality checks (optional)
```powershell
cd D:\AskPolicy\backend
uv run pytest
uv run ruff check .
uv run ruff format --check .

cd D:\AskPolicy\frontend
npm run lint
npm run build
```

## 4. How to insert data into the vector database (Qdrant)

**Insertion is done by the ingestion pipeline** (there is no browser upload). The pipeline embeds chunks locally and stores them in Qdrant Cloud.

1. Add, edit, or remove files under `documents/` (e.g. `documents/hr/leave_policy.pdf`, `documents/travel.txt`).
2. Run:
   ```powershell
   cd D:\AskPolicy\backend
   uv run python -m ingestion.ingest
   ```
3. What happens safely:
   - Discovers supported files and extracts/chunks/embeds them.
   - Creates a **temporary physical collection**, uploads vectors + payload (`page_content`, `metadata.source/file_name/file_type/chunk_index`, and `page_number` for PDFs), and validates count/query.
   - **Atomically moves the alias** (`QDRANT_COLLECTION`, default `askpolicy_documents`) to the new collection.
   - Deletes the previous physical collection and orphaned temporary ingestion collections **only after** success.
4. Verify: call `GET /api/documents` or use the sidebar refresh in the UI. New questions will retrieve from the updated alias immediately.

**Important:** Each ingestion run indexes the *current* contents of `documents/` as the full knowledge base (it is not incremental append). To "add" a document, keep existing files in `documents/` and add the new one, then re-run ingestion.

## 5. How to delete/replace previous data in the vector database

### 5.1 Replace indexed data with a new set (recommended)
This is the normal way to remove old documents from search results:

1. Update `documents/` to contain exactly the files you want indexed (remove files you no longer want).
2. Run ingestion again:
   ```powershell
   cd D:\AskPolicy\backend
   uv run python -m ingestion.ingest
   ```
3. On success, the alias points to a new collection containing only the new documents; the previous physical collection is deleted automatically. Previous documents will no longer be searchable.

> If ingestion fails, the **previous index is kept** active (no partial destructive replacement).

### 5.2 If there are no valid documents
If `documents/` has no supported files with extractable text, ingestion raises a "no documents found" condition and **does not erase** the existing Qdrant index. This is by design to prevent accidental data loss.

### 5.3 Fully remove/clear the Qdrant index (delete alias and collection)
Use this only when you intentionally want no indexed data in Qdrant (e.g. reset the environment). **This is destructive** for the configured collection/alias.

From `backend/` (uses credentials from `.env`):

```powershell
cd D:\AskPolicy\backend
uv run python -c "
from app.core.config import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models as m

alias = settings.qdrant_collection
client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=settings.qdrant_timeout_seconds)

targets = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
target = targets.get(alias)

if target:
    client.update_collection_aliases(change_aliases_operations=[
        m.DeleteAliasOperation(delete_alias=m.DeleteAlias(alias_name=alias))
    ])
    client.delete_collection(collection_name=target)
    print(f'Deleted alias {alias} and collection {target}')
else:
    print(f'Alias {alias} not found; nothing deleted')

# Optional: remove leftover temporary ingestion collections for this alias
for col in client.get_collections().collections:
    if col.name.startswith(alias + '__ingest_'):
        client.delete_collection(collection_name=col.name)
        print(f'Deleted leftover temp collection {col.name}')
"
```

After clearing:
- `GET /api/health` will show the collection as not ready / degraded.
- `GET /api/documents` will report that no documents are indexed (503 until ingested).
- To use the app again, add documents to `documents/` and run `uv run python -m ingestion.ingest`.

**Safety notes:** double-check `QDRANT_COLLECTION` in `.env` before running; this removes data for that alias in the connected Qdrant cluster. It does not delete local files in `documents/`.

## 6. Troubleshooting

- **"Qdrant is not configured" / health degraded:** verify `QDRANT_URL` and `QDRANT_API_KEY` in `.env`.
- **Chat fails with Groq error:** verify `GROQ_API_KEY` and `GROQ_MODEL` are valid and supported.
- **No documents indexed:** ensure files are in `documents/` with supported extensions and contain extractable text, then re-run ingestion.
- **PDF text missing:** scanned/image-only PDFs and some encrypted PDFs may not extract; use text-based PDFs or convert/OCR externally.
- **Changes not reflected:** re-run ingestion and use the UI refresh; backend restart is not required for re-indexing (restart is required for code/config changes depending on setup).
- **Conversation memory resets:** memory is in-process (`InMemorySaver`); restarting the backend or running multiple workers clears/isolates threads.
- **Embedding model download slow/fails:** ensure network access on first ingestion; model files are cached in `backend/.cache/huggingface/`.
