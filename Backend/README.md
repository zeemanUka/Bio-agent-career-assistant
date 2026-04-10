# Bio Agent Backend

Python API service for the Bio Agent career assistant.

## What changed

- Gradio has been removed from the backend.
- The backend now exposes HTTP endpoints for a separate frontend.
- The core agent logic still lives in `agent.py`.

## Endpoints

- `GET /api/health`
- `POST /api/chat`
- `POST /api/chat/stream`

Example chat request:

```json
{
  "message": "What are your core technical strengths?",
  "history": [
    { "role": "user", "content": "Hi" },
    { "role": "assistant", "content": "Hello" }
  ]
}
```

Example chat response:

```json
{
  "message": "..."
}
```

## Local setup

1. Create and use a project-local virtual environment in `Backend/.venv`.
2. Copy `.env.example` to `.env`.
3. Set `LLM_API_BASE_URL`, `LLM_API_KEY`, and `AGENT_MODEL`.
4. Start the API from the `Backend/` directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --reload
```

The API listens on `http://localhost:8000` by default.

## Frontend integration

Set `CORS_ALLOW_ORIGINS` to the frontend origin. The default is:

```env
CORS_ALLOW_ORIGINS=http://localhost:3000
CHROMA_ANONYMIZED_TELEMETRY=false
CHROMA_PRODUCT_TELEMETRY_IMPL=chroma_telemetry.NoOpProductTelemetryClient
```

## Knowledge base files

Add `.txt` or `.pdf` files under `knowledge/`. The backend builds its Chroma index on first run and stores runtime data in:

- `chroma_runtime/`
- `db_runtime/`
