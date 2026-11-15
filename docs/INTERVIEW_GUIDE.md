# AskPolicy interview guide

This guide explains the project in the terms an AI engineering interviewer is
likely to use. It is based on the actual implementation, not a generic RAG
tutorial. Each section ends with the detail you should be able to point to in the
code.

## 60-second overview

AskPolicy answers employee questions about company policy documents. Documents
are ingested offline: text is extracted from PDF, TXT, and DOCX files, split into
overlapping chunks, embedded locally with `all-MiniLM-L6-v2`, and stored in
Qdrant Cloud behind a collection alias. At question time, the FastAPI backend
embeds the question, retrieves the five most similar chunks, optionally filters
by source document, and sends only those excerpts to a Groq-hosted LLM with a
strict grounding prompt. The response includes the answer plus the real chunks
used as sources. LangGraph keeps the last five question/answer exchanges in
memory per thread. If retrieval finds nothing, the backend returns an explicit
not-found message without calling the LLM.

## 1. What is RAG?

Retrieval-Augmented Generation means combining a search step with a language
model. Instead of asking the model to answer from its training data, the system
first retrieves relevant text from a private corpus, then gives that text to the
model as context for the answer.

In AskPolicy the retrieval step is a dense vector search in Qdrant, and the
generation step is a Groq chat completion.

**Where:** `RagService._answer_node()` calls `RetrievalService.retrieve()` and
then `LLMService.generate_answer()`.

## 2. Why use RAG here?

Three reasons:

1. **Private knowledge.** Company policies are not in the model's training data.
2. **Freshness.** Re-running ingestion updates the answers without retraining or
   redeploying a model.
3. **Verifiability.** The system can show exactly which document chunks supported
   an answer.

RAG also reduces hallucination, but does not eliminate it. AskPolicy adds
explicit grounding rules, a not-found path, and source citations on top of
retrieval.

**Where:** `app/core/prompts.py`, `app/services/rag_service.py`,
`frontend/src/components/SourceReferences.tsx`.

## 3. What are embeddings?

An embedding is a fixed-length numeric vector that represents the meaning of
text. Texts with similar meaning end up close together in vector space, even if
they do not share exact keywords. A question like "How much vacation do I get?"
can be close to a passage about "annual leave entitlement" without either
containing the other's wording.

AskPolicy embeds both document chunks and user questions with the same model so
they share one vector space.

**Where:** `EmbeddingService.embed_documents()` and `embed_query()`.

## 4. How does all-MiniLM-L6-v2 work at a high level?

It is a sentence-transformer model: a small transformer encoder trained to map
sentences and paragraphs to vectors such that semantically similar pairs are
close. Input text is tokenized into word pieces, passed through transformer
layers, and pooled into one 384-dimensional vector. AskPolicy additionally
normalizes vectors, which is why cosine similarity and dot product produce the
same ranking.

It runs on CPU, is small enough for a laptop, and is used locally — no embedding
API call.

**Where:** `app/services/embedding_service.py`; normalization is configured with
`normalize_embeddings=True`.

## 5. Why is the embedding dimension 384?

The dimension is a property of the model architecture. `all-MiniLM-L6-v2` uses a
384-dimensional hidden representation, so every text it embeds becomes 384
numbers. The Qdrant collection must be created with exactly that size; otherwise
uploads fail or similarity is meaningless.

AskPolicy does not hardcode the assumption blindly. During ingestion,
`verify_dimension(384)` embeds a probe string and fails before creating the
collection if the model returns a different size.

**Where:** `Settings.embedding_dimension`, `EmbeddingService.verify_dimension()`,
and `_create_collection()` in `ingestion/ingest.py`.

## 6. What is chunking, and why is it needed?

A whole document is usually too large to embed usefully and too large to fit
into a prompt. Chunking splits documents into smaller passages. Each chunk gets
its own embedding, so retrieval can return the specific passage that answers a
question instead of an entire handbook.

Chunks also need metadata so the UI can say which file and page an answer came
from.

**Where:** `ingestion/chunker.py`.

## 7. How does recursive character splitting work?

`RecursiveCharacterTextSplitter` tries a hierarchy of separators:

```python
["\n\n", "\n", " ", ""]
```

It first tries to split on paragraph breaks. If a piece is still too large, it
tries line breaks, then spaces, and finally splits inside a word as a last
resort. This preserves readable boundaries far better than cutting every N
characters.

AskPolicy uses `chunk_size=1000` and `chunk_overlap=200` characters and sets
`keep_separator=True`.

**Where:** `build_splitter()` and `chunk_documents()`.

## 8. Why use overlap?

Overlap duplicates a small amount of text between consecutive chunks. If a rule,
definition, or table header spans a chunk boundary, the complete thought appears
in at least one chunk. Without overlap, retrieval could return half a sentence
and lose the condition that makes it meaningful.

The tradeoff is storage and duplicate retrieval. AskPolicy keeps overlap at 200
characters, well below the 1000-character chunk size.

**Where:** `DEFAULT_SEPARATORS` and `build_splitter()`.

## 9. What is a vector database?

A vector database stores embeddings and supports nearest-neighbor search over
them at scale. Unlike a relational database, the primary query is "which vectors
are closest to this query vector?" Qdrant also stores payloads — arbitrary JSON
attached to each vector — which AskPolicy uses for text and metadata.

**Where:** Qdrant Cloud, accessed through `QdrantService` and LangChain's
`QdrantVectorStore`.

## 10. How does Qdrant store vectors and metadata?

Each Qdrant point has:

- an id (AskPolicy uses a UUID string),
- a 384-dimensional vector,
- a payload.

AskPolicy's payload follows LangChain's convention:

```json
{
  "page_content": "the exact chunk text",
  "metadata": {
    "source": "hr/leave_policy.pdf",
    "file_name": "leave_policy.pdf",
    "file_type": "pdf",
    "page_number": 2,
    "chunk_index": 7
  }
}
```

Collections are created with cosine distance and the verified vector size.
There is no separate metadata database: `GET /api/documents` scrolls payloads and
aggregates counts per source.

**Where:** `_upsert_chunks()` in `ingestion/ingest.py` and
`QdrantService.get_index_summary()`.

## 11. What does cosine similarity mean?

Cosine similarity measures the angle between two vectors:

```text
cosine(a, b) = (a · b) / (|a| * |b|)
```

It ranges from -1 to 1 in general; for normalized embeddings, higher means more
semantically similar. It cares about direction, not magnitude, which suits text
embeddings. Because AskPolicy normalizes embeddings, ranking by cosine and by dot
product is equivalent.

**Where:** Qdrant collection distance is `Distance.COSINE`; LangChain returns a
score for each result.

## 12. How does dense vector search work?

1. The question is embedded into the same 384-dimensional space.
2. Qdrant compares it with stored vectors and returns the nearest `top_k`.
3. AskPolicy maps the results to `RetrievedChunk` objects with content and
   metadata.

The default `top_k` is 5 and is configurable with `RETRIEVAL_TOP_K`. This is
"dense" retrieval because it uses the embedding vector, not keyword overlap.

**Where:** `RetrievalService.retrieve()`.

## 13. What does metadata filtering do?

Filtering restricts search to a subset of points before the top-k limit is
applied. AskPolicy supports filtering by `metadata.source`, which is the relative
path inside `documents/`.

For example, selecting `hr/leave_policy.pdf` in the sidebar sends that path as
the `source` field. RetrievalService builds a Qdrant `Filter` with a
`FieldCondition` on `metadata.source`. Qdrant then searches only those points.

This is better than filtering results in Python after retrieval, because a
filtered query still gets five relevant chunks instead of whatever survives from
a global top five.

**Where:** `RetrievalService._build_filter()`.

## 14. How does Groq generate responses?

Groq hosts open models and serves chat completions through an OpenAI-compatible
API. AskPolicy uses LangChain's `ChatGroq` with a configurable model name,
temperature 0, a token limit, a timeout, and two retries.

The service sends a system message containing the grounding rules and one user
message containing conversation context, numbered excerpts, and the question. It
then normalizes the response content to plain text.

**Where:** `LLMService.generate_answer()`.

## 15. How does retrieved context reach the LLM?

`build_user_prompt()` creates one user message with three parts:

1. Recent conversation, labelled as unverified context.
2. Numbered excerpts, each fenced with `--- BEGIN SOURCE [n] ---` and
   `--- END SOURCE [n] ---`, including filename, path, page, and chunk index.
3. The employee question and a reminder to answer only from the excerpts.

Conversation history is not replayed as separate chat turns. Putting it inside
the user message makes the trust boundary explicit: the model sees that prior
answers are context, not evidence.

**Where:** `app/core/prompts.py`, called from `RagService._answer_node()`.

## 16. How are source citations generated?

Citations are not generated by parsing the model's prose. The backend already
knows which chunks it retrieved, so it returns those chunks as structured
`SourceReference` objects alongside the answer. The model is asked to cite
excerpt numbers such as `[1]`, and the backend parses those markers to return
only the chunks the answer actually cited. If no citation markers are present,
all retrieved chunks are returned as a fallback.

If the model returns the exact not-found message, the backend deliberately
returns an empty source list so unrelated chunks are not shown as if they
supported a refusal.

**Where:** `RagService._to_source()` and `app/schemas/chat.py`.

## 17. What happens when no relevant answer exists?

There are two paths:

1. **Nothing retrieved.** If Qdrant returns no chunks (empty index or a filter
   with no matches), the backend returns
   `"I couldn't find this information in the available company documents."`
   and does not call the LLM at all.
2. **Retrieved but insufficient.** If the model decides the excerpts do not
   answer the question, the system prompt requires the same exact sentence. The
   backend detects it and clears the sources.

This design makes refusal a first-class behavior rather than an accidental model
output.

**Where:** `NOT_FOUND_MESSAGE`, `_answer_node()`, and the system prompt rules.

## 18. Walk through the complete ingestion workflow

1. `run_ingestion()` resolves `documents/` and the configured alias.
2. `discover_documents()` finds supported files recursively.
3. `load_document()` extracts text per PDF page or per TXT/DOCX file and builds
   metadata from the real relative path.
4. Empty files are skipped; unreadable supported files abort the run.
5. `chunk_documents()` splits text with overlap and assigns per-file chunk
   indexes, preserving page numbers only for PDFs.
6. `EmbeddingService` verifies 384 dimensions and embeds all chunks locally.
7. A temporary collection is created with cosine distance.
8. Points are uploaded in batches of 64 with `page_content` and metadata.
9. The point count is checked exactly and a probe query confirms searchability.
10. The alias is atomically moved to the new collection.
11. The previous collection and orphaned temporary collections are deleted.
12. Real statistics are printed, including discovered/processed/skipped files,
    PDF pages, chunk count, and whether a previous index was replaced.

If anything fails before step 10, the old alias and collection remain intact.

**Where:** `ingestion/ingest.py`.

## 19. Walk through the complete question-answering workflow

1. React sends `POST /api/chat` with the question, optional source filter,
   optional thread id, and recent session history.
2. Pydantic validates a non-empty, length-limited question and optional fields.
3. `RagService.chat()` resolves or creates a thread id.
4. If the thread has no state but history was supplied, it seeds the last five
   exchanges once.
5. LangGraph loads the thread's messages and invokes the `answer` node with the
   new human message.
6. The node embeds the current question and retrieves top-k chunks, optionally
   filtered by source.
7. With no chunks: return the not-found message, no LLM call, no sources.
8. With chunks: build the grounded prompt and call Groq.
9. Map retrieved chunks to source objects (or clear them on refusal).
10. Append the answer to memory and trim to five exchanges.
11. Return `{answer, sources, thread_id}`.
12. React renders Markdown, source cards, and stores the transcript in
    `sessionStorage`.

**Where:** `app/api/routes/chat.py`, `app/services/rag_service.py`,
`frontend/src/hooks/useChat.ts`.

## 20. Why this architecture?

- **Separate ingestion and serving.** Indexing is expensive and occasional;
  question answering must stay fast. The CLI can rebuild the index while the API
  keeps serving the old one.
- **Alias-based replacement.** It gives near-atomic updates without a
  destructive delete-first step.
- **Local embeddings.** No per-query embedding cost, no external dependency, and
  policy text never leaves the machine for embedding.
- **Qdrant Cloud.** Managed vector search with filtering, without running
  infrastructure.
- **Groq.** Fast hosted inference with a configurable model, keeping generation
  separate from retrieval.
- **One-node LangGraph.** Conversation memory is a real requirement; using
  `MessagesState` plus a checkpointer avoids inventing custom session storage.
  There are no agents or tools, so the architecture stays a RAG pipeline.
- **Thin routes, focused services.** Each concern can be tested and explained
  independently.

## Likely follow-up questions

### Why not fine-tune a model instead?

Fine-tuning changes model behavior and style but is a poor way to store changing
policy facts. It is expensive, hard to update, and still cannot cite sources.
RAG keeps the knowledge external, updatable, and auditable. Fine-tuning could
later complement RAG for tone or domain jargon, but it would not replace
retrieval here.

### Why not use an agent or tool-calling framework?

The task is one retrieval step followed by one generation step. An agent adds
branching, latency, cost, and failure modes without improving this workflow.
LangGraph is used only for thread memory, which is an explicit requirement.

### How do you reduce hallucination?

Layered defenses:

- retrieve only relevant chunks;
- pass only those chunks as evidence;
- instruct the model to refuse when evidence is insufficient;
- skip the LLM entirely when retrieval is empty;
- clear sources on refusal;
- label history as unverified;
- treat document text as untrusted to resist prompt injection;
- show sources so a human can verify the answer.

None of these is a formal guarantee. Evaluation and human review are still
needed.

### Why chunk at 1000 characters instead of tokens?

Characters are model-independent, simple, and cheap. For normal English, 1000
characters is roughly 200–250 tokens, under MiniLM's 256-token limit. The
tradeoff is that dense text can be truncated. A production system would use the
model tokenizer or a token-aware splitter, and would evaluate chunk sizes on a
labelled question set.

### Why top_k = 5?

It is a reasonable default balancing context size, latency, and recall. Too few
chunks can miss the answer; too many add noise and cost. It is configurable so it
can be tuned with retrieval evaluation.

### What happens if the backend restarts mid-conversation?

In-memory LangGraph state is lost. The browser still has the transcript and
thread id in `sessionStorage`, and the next request includes recent history. The
backend seeds a new thread from that history once. This restores conversational
context but does not restore the exact server state, and retrieval still uses the
current question only.

### Why does retrieval not use conversation history?

This is a known limitation, not an oversight. Retrieval embeds the current
question verbatim. Follow-ups like "what about part-time staff?" can retrieve
weaker evidence than a self-contained question. The standard fix is query
rewriting: ask a small LLM to turn the follow-up plus history into a standalone
search query. That was deliberately left out of Version 1 to keep the pipeline
transparent and avoid an extra model call.

### How does document replacement avoid downtime?

Ingestion writes to a new physical collection and validates it. Only then does
it delete and recreate the alias in a single Qdrant request. Queries use the
alias, so they see either the old target or the new one. If the switch fails, the
code tries to restore the old alias and deletes the temporary collection.

### What if two ingestion runs happen at once?

Each run uses a unique temporary collection name, so uploads do not collide. The
alias switch is last-writer-wins, and cleanup protects collections referenced by
aliases. Running two ingests concurrently is not a supported operational
workflow, but it will not corrupt payloads. A production system would add a lock
or a job queue.

### How would you evaluate this system?

Build a labelled set of policy questions with expected source chunks and
answers. Measure retrieval recall@k and MRR separately from answer correctness,
faithfulness, and refusal accuracy. RAGAS or a custom LLM-judge can help, but
human review is needed for policy accuracy. The current repository has unit and
integration-style tests, not an offline quality benchmark.

### How would you scale it?

- Replace `InMemorySaver` with a shared checkpointer if running multiple workers.
- Add query rewriting and hybrid retrieval for harder questions.
- Add reranking and token-aware chunking.
- Cache embeddings for repeated questions.
- Add authentication and per-document access control.
- Add ingestion locking and incremental indexing.

### What are the main failure modes?

- Qdrant unreachable or alias missing → `503` with a safe message.
- Groq unavailable or empty response → `502`.
- Scanned PDF with no text layer → file skipped; if nothing remains, ingestion
  refuses to replace the index.
- Embedding dimension mismatch → ingestion fails before touching the collection.
- Follow-up question lacking context → weaker retrieval.
- Model ignores instructions → citations still reflect retrieved chunks, but the
  answer must be verified by a human.

## Demo script

1. Show the empty `documents/` folder and explain that private policies are not
   committed.
2. Add a PDF, a DOCX, and a nested TXT file.
3. Run `uv run python -m ingestion.ingest` and point out real counts and the alias
   replacement message.
4. Start the backend and frontend.
5. Ask a question answered by one document, then show the source card and page
   number.
6. Select that document in the sidebar and ask a scoped question.
7. Ask something absent from the documents and show the not-found response with
   no sources.
8. Ask a follow-up, then clear the chat and explain `POST /api/chat/reset`.
9. Replace the documents, re-ingest without restarting the backend, refresh the
   sidebar, and show the new index.
10. Run `uv run pytest`, `uv run ruff check .`, `npm run lint`, and
    `npm run build`.

## One-sentence defensible summary

AskPolicy is a deliberately simple, grounded RAG system: local MiniLM embeddings,
Qdrant cosine retrieval with metadata filtering, a strict Groq prompt with
explicit refusal behavior, safe alias-based re-indexing, and process-local
LangGraph memory for short follow-up conversations.
