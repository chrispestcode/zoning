# Code Review — Zoning RAG Pipeline

**Date:** 2026-11-04
**Scope:** every Python file under `src/`, plus the SQLite database and git setup.
**Not re-flagged:** two choices already made on purpose — QA pairs drawn from a random cross-document sample, and fault tolerance for partly generated documents being deferred.

## Summary

| Severity | Count | What it means |
|---|---|---|
| Critical | 2 | Broken right now; crashes or returns nothing |
| High | 4 | Runs, but produces wrong or misleading results |
| Medium | 11 | Hurts quality, performance, or reliability |
| Low | — | Cleanup and tidiness |

The two outright breaks are the search API and `evals.py generate`. The most damaging problem is in chunking: it lowers retrieval quality across the whole system.

---

## Critical — broken right now

### 1. The search API never returns results
**File:** `src/backend.py:26`

**The problem.** `query_qdrant` returns a *list* on success and a *dict* (`{"results": []}`) on failure, but the endpoint only proceeds when it receives a dict. So:

- a successful search returns an empty response;
- a failed search crashes on `qdrant_results[:20]`, because a dict can't be sliced.

**It goes further:**

- The endpoint reads `.DocumentId` and `.ChunkText` from Qdrant points, which only have `.payload`. The chunks should come from `qdrant_manager.get_results(...)`.
- `from src.content_generator` uses a different import style from every other file, which uses bare imports (`from qdrant import ...`). The server only starts if both the repo root and `src` are on the path.
- `src/frontend.py` expects `text_response` to be a string but receives a dict, and it crashes on `len(query_forms)` when that key is missing.

**The fix.** Have `query_qdrant` always return a list (`[]` on failure). In the endpoint, check for a non-empty list, load chunks with `get_results`, and pass the answer string through to the frontend.

### 2. `evals.py generate` crashes
~~**File:** `src/evals/evals.py:315`~~

~~**The problem.** The flag is defined as `--douments`, so `args.documents` raises `AttributeError`.~~

~~**The fix.** Rename it to `--documents`.~~

---

## High — runs, but results are wrong

### 3. Chunking splits text at almost every sentence
**File:** `src/chunker.py:35`

**The problem.** With `breakpoint_threshold_type='percentile'`, `breakpoint_threshold_amount` is on a 0–100 scale (the default is 95). The code passes `0.95`, which means "split wherever the distance is above the 1st percentile", so nearly every sentence boundary becomes a split.

**Evidence from the database:**

| Measure | Value |
|---|---|
| Total chunks | 18,899 |
| Average length | 376 characters |
| Chunks under 300 characters | 11,188 (59%) |

This is probably why `get_results` pads each hit with neighboring chunks.

**The fix.** Set the value to `95`.

**The catch.** Every document then has to be re-chunked, re-embedded and re-upserted to Qdrant. The current eval sets become stale too, because their chunk and derivative IDs will change. TRACK chunk data like chart above.

### 4. A model failure in `get_chunk_response` looks like a real answer
**File:** `src/content_generator.py:106`

**The problem.** `get_derivatives_chunks` and `get_qa_pairs` both strip a ```` ```json ```` fence before parsing; `get_chunk_response` doesn't. When Claude wraps its JSON that way, `json.loads` fails and the method returns `"An error occurred while generating the response."` *as the answer*.

**Why it matters.** Generation evals would score that error text as if it were a real answer, and the frontend would show it to users.

**The fix.**

- Strip the fences the same way the other two methods do.
- On failure, return something the caller can tell apart from a real answer, or raise.

### 5. The Claude backend ignores `max_tokens` and `temperature`
**File:** `src/llm/claude_backend.py:46-51`

**The problem.** `max_tokens` is read from `kwargs`, but the API call hardcodes `1024`. `temperature` is never passed at all.

**Why it matters.**

- The temperature settings (0.85 for derivatives and answers, 0.34 for QA) have no effect.
- Ten QA pairs in JSON can run past 1,024 tokens. The cut-off JSON then fails to parse, and `get_qa_pairs` quietly returns `[]`.

**The fix.** Pass `max_tokens=max_tokens` and forward `temperature` from `kwargs`.

### 6. Cloud mode deletes downloaded files before parsing them
**File:** `src/main.py:55-69`

**The problem.** The S3 files are downloaded into a `TemporaryDirectory`, which is deleted when its `with` block ends. `parse_files` runs after that block, so the files are already gone.

**Impact today:** none, because `cloud_storage_enabled = False`. It will break as soon as cloud mode is turned on.

**The fix.** Call `parse_files(document_files)` inside the `with TempDirectory()` block.

---

## Medium

### 7. Ollama backend
**File:** `src/llm/ollama_backend.py:46-55`

- It calls blocking `requests.post` inside an `async` method, so nothing runs concurrently. Use `httpx.AsyncClient`, or wrap the call in `asyncio.to_thread`.
- Ollama ignores a top-level `temperature`; it belongs under `"options": {"temperature": ...}`.
- `do_sample` isn't an Ollama parameter.

### 8. Retrieval quality
**Files:** `src/qdrant.py`, `src/pipeline/chunker.py`

- **No BGE query prefix.** BGE recommends prefixing *queries* (not documents) with `"Represent this sentence for searching relevant passages: "`. The code doesn't.
- **Silent truncation.** 192 chunks are longer than BGE's 512-token limit and are cut off without any warning.
- **Lopsided neighbor window.** `range(-3, 3)` in `get_results` takes 3 chunks before each hit but only 2 after, and it can pull in chunks from a different document.

### 9. `query_qdrant` returns two different types
**File:** `src/qdrant.py:244`

It returns a list on success and a dict on failure. Returning `[]` on failure removes the root cause of #1 and the `isinstance` workaround in `evals.py`.

### 10. Generation evals are hard to interpret
**File:** `src/evals/evals.py`

- **Perfect-retrieval context.** The context passed in is the ground-truth chunks, not what retrieval actually returned. That measures answer quality under perfect retrieval, which is fine if intended, but label it that way.
- **No judge model.** ragas defaults to OpenAI as its judge, and none is configured.
- **Broken import.** ragas still can't be imported in this environment: ragas 0.4.3 needs a module that `langchain-community` 0.4 removed.

### 11. A crash between steps leaves a document half-processed
**File:** `src/chunker.py:62-67`

Chunks are inserted, and the document status is updated, in separate transactions. If the run dies between them, the document stays `NOT_STARTED` and is chunked again on the next run, which creates duplicate chunks.

### 12. The parser reloads its models for every PDF
**File:** `src/parser.py:73`

A new `DocumentConverter` is built for each file, reloading docling's models every time. Build it once in `__init__`.

### 13. The databases aren't git-ignored
`src/document_database.db` and `src/document_database.backup-2026-10-28.db` are about 300 MB each. A `git add .` would try to commit them, and GitHub rejects files over 100 MB.

**The fix.** Add `*.db` to `.gitignore`.

### 14. `update_*_status` methods report success without checking
**File:** `src/document_database.py:947-1078`

They return every ID that was passed in, even when no row matched. Check `cursor.rowcount` before counting an ID as updated.

### 15. `get_unique_chunk_ids_from_derivatives` is broken
**File:** `src/document_database.py:698`

It queries a `Derivatives` table and a `CHUNK_ID` column that don't exist (the real names are `Derivative` and `ChunkId`), and it reads `row[1]` from a one-column result. Nothing calls it yet.

### 16. The PDF crawler has no limits
**File:** `src/scripts/web_crawler/web_crawler_pdf_with_logging.py`

It follows off-site links up to depth 3 and ignores robots.txt. Its start URL is a Department of Labor ACA page, apparently left over from another project.

### 17. Imports depend on `PYTHONPATH`
`src` isn't an installed package, so imports only work when `src` is put on the path by hand (PyCharm's Sources Root, or `PYTHONPATH=.`). This is also behind the mixed imports in #1. The long-term fix is a real package (for example `src/zoning/`) that uv installs into the environment.

---

## Low — cleanup

### Hardcoded machine-specific values
- `/home/tchrisp/...` paths in `src/parser.py:21` and `src/parser.py:170`.
- The AWS role ARN, including the account ID, in `src/main.py` and `src/aws/aws_session.py`. Move it to `.env` or `config.yaml`.

### Dead or broken code
- `Embedder` loads a second copy of BGE through `AutoModel` (`src/embedding.py:26-27`) that it never uses. That copy costs about 1.3 GB of memory.
- `generate_text_embeddings` and `generate_image_embeddings_clip` in `src/pipeline/embedding.py` call methods and attributes that don't exist.
- `generate_content` in `src/main.py` calls `get_chunk_table`, which doesn't exist.
- Both benchmark scripts call an undefined `search_and_print_results`.
- The gzip helpers in `src/utils/utils.py` are unused, while the database uses zlib.
- `_evals_log_dir` (`src/evals/evals.py:155`) reads a `paths.logs.evals` key that isn't in the config.
- `src/evals/evals.py` has an unused `MagicMock` import.
- `src/test.py` is a leftover experiment.

### Logging
- Insert errors log the entire record. For example, `Failed to insert document {document}` dumps the whole parsed document.
- `src/utils/cuda_logging.py` configures logging when it's imported and writes `cuda_memory.log` to whatever folder you run from.

### Performance nits
- Inserts commit once per row; one commit per batch is much faster.
- `query_qdrant` uses `with_vectors=True`, fetching 1,024 floats per hit that nothing reads.
- `query_qdrant` makes three collection-status calls per query.
- `QdrantManager`'s default argument `qdrant_url=get_qdrant_host()` reads the config at import time.

### Small correctness issues
- `"question": {question}` in `evaluate` builds a set, not a string.
- SQLite connections are never closed: `with conn` commits but doesn't close.
- The parser's device check ignores Apple's MPS GPU.
- The `src/models/contracts.py` docstring describes return types that have since changed.

---

## Suggested order

1. **#2 and #4.** Both are one-line fixes; they unblock eval generation and make its results trustworthy.
2. **#5.** Prompts then get the temperature and token limit they ask for.
3. **#1 and #9.** These get the app working end to end.
4. **#3, last.** It means rebuilding the index. Afterwards, rerun the evals and compare against the current baseline:

| Metric | Current |
|---|---|
| MAP | 0.544 |
| MRR | 0.614 |
