# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A full-stack RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. Users interact via a chat UI; the backend uses semantic search over indexed course documents to retrieve relevant context, which is passed to Claude to generate accurate, grounded answers.

**Stack:** Python, FastAPI, ChromaDB, `sentence-transformers`, Anthropic Claude, vanilla JS frontend.

## Commands

```bash
# Install dependencies
uv sync

# Run the application (starts uvicorn on port 8000)
./run.sh

# Or manually from the backend directory
cd backend
uv run uvicorn app:app --reload --port 8000
```

The app serves at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

Requires `ANTHROPIC_API_KEY` in a `.env` file at the project root (see `.env.example`).

## Architecture

This is a full-stack RAG chatbot. The **backend** (`backend/`) is a FastAPI app served via uvicorn. The **frontend** (`frontend/`) is plain HTML/CSS/JS, served as static files by FastAPI itself — there is no separate frontend dev server.

### Backend components and their roles

| File | Role |
|---|---|
| `app.py` | FastAPI entrypoint. Two API routes (`POST /api/query`, `GET /api/courses`). On startup, loads docs from `../docs/`. Mounts `../frontend/` as static files. |
| `rag_system.py` | Central orchestrator. Wires all components together and exposes `query()` and `add_course_folder()`. |
| `ai_generator.py` | Wraps the Anthropic SDK. Makes up to two Claude API calls per query: one to decide whether to search, one to generate a final answer after tool results. |
| `search_tools.py` | Defines `CourseSearchTool` (an Anthropic tool Claude can invoke) and `ToolManager`. Tools follow an abstract `Tool` base class — new tools can be registered via `ToolManager.register_tool()`. |
| `vector_store.py` | ChromaDB wrapper with two collections: `course_catalog` (course-level metadata) and `course_content` (text chunks). Embeddings use `sentence-transformers`. |
| `document_processor.py` | Parses `.txt`/`.pdf`/`.docx` course files into `Course` + `CourseChunk` objects, and splits content into overlapping chunks. |
| `session_manager.py` | In-memory conversation history keyed by session ID, capped at `MAX_HISTORY` exchanges. |
| `models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk`. |
| `config.py` | Single `Config` dataclass loaded from env vars. Key values: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, `MAX_RESULTS=5`, `MAX_HISTORY=2`, `EMBEDDING_MODEL=all-MiniLM-L6-v2`. |

### Query flow

1. Frontend POSTs `{ query, session_id }` to `/api/query`
2. `RAGSystem.query()` fetches conversation history and calls `AIGenerator.generate_response()`
3. **1st Claude call** — Claude decides whether to invoke `search_course_content` tool
4. If tool is called: `VectorStore.search()` performs semantic search in ChromaDB (with optional course/lesson filters), results are returned to Claude
5. **2nd Claude call** — Claude generates a final answer from the search results (no tools this time)
6. Response + sources + session_id are returned to the frontend

### Course document format

Files in `docs/` must follow this structure for the parser to extract metadata correctly:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <title>
...
```

ChromaDB is persisted locally at `backend/chroma_db/`. On startup, already-indexed courses are skipped (deduplication by title).
