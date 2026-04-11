"""
Bio Agent API
-------------
HTTP entrypoint for the backend service.
"""

import json
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent import BioAgent
from config import API_HOST, API_PORT, CHROMA_USE_CLOUD, CORS_ALLOW_ORIGINS


class ChatMessage(BaseModel):
    """A single chat message exchanged with the frontend."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response body returned to the frontend."""

    message: str


class HealthResponse(BaseModel):
    """Basic health information for the backend."""

    status: str
    cors_origins: list[str]
    chroma_mode: Literal["cloud", "local"]
    chroma_cloud_configured: bool


agent: BioAgent | None = None


def _get_agent() -> BioAgent:
    """Create the singleton agent lazily and reuse it across requests."""
    global agent
    if agent is None:
        agent = BioAgent()
    return agent


def _runtime_error_message(exc: Exception) -> str:
    """Return a user-visible fallback message instead of a raw 500."""
    return (
        "The assistant backend hit a runtime error while handling this request. "
        "Please retry in a moment. "
        f"Runtime error: {type(exc).__name__}: {exc}"
    )


app = FastAPI(
    title="Bio Agent API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Expose a simple readiness endpoint for local dev and deployment."""
    return HealthResponse(
        status="ok",
        cors_origins=CORS_ALLOW_ORIGINS,
        chroma_mode="cloud" if CHROMA_USE_CLOUD else "local",
        chroma_cloud_configured=CHROMA_USE_CLOUD,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Accept a user message and return the assistant response."""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message must not be empty.")

    history = [{"role": item.role, "content": item.content} for item in request.history]
    try:
        answer = _get_agent().chat(message=message, history=history)
    except Exception as exc:
        answer = _runtime_error_message(exc)
    return ChatResponse(message=answer)


def _sse_event(event: str, payload: dict) -> str:
    """Encode a server-sent event payload."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream chat deltas for incremental frontend rendering."""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message must not be empty.")

    history = [{"role": item.role, "content": item.content} for item in request.history]

    def event_stream():
        answer_parts: list[str] = []

        try:
            for delta in _get_agent().stream_chat(message=message, history=history):
                answer_parts.append(delta)
                yield _sse_event("message", {"delta": delta})
        except Exception as exc:
            fallback = _runtime_error_message(exc)
            answer_parts.append(fallback)
            yield _sse_event("message", {"delta": fallback})

        yield _sse_event("done", {"answer": "".join(answer_parts)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)
