"""
Bio Agent Configuration
-----------------------
Single source of truth for all paths, model names, and thresholds.
"""

import os

from dotenv import load_dotenv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, ".env"))


def _get_int_env(name: str, default: int) -> int:
    """Read an integer env var with a safe fallback."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    """Read a boolean env var with common truthy and falsy forms."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _get_csv_env(name: str, default: list[str]) -> list[str]:
    """Read a comma-separated env var into a list of non-empty strings."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def is_local_base_url(url: str) -> bool:
    """Detect whether the LLM endpoint points to a loopback host."""
    prefixes = (
        "http://localhost",
        "https://localhost",
        "http://127.0.0.1",
        "https://127.0.0.1",
    )
    return url.startswith(prefixes)


def _is_running_on_vercel() -> bool:
    """Detect whether the backend is executing inside a Vercel runtime."""
    return bool(
        os.getenv("VERCEL")
        or os.getenv("VERCEL_ENV")
        or os.getenv("VERCEL_URL")
    )


# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = _THIS_DIR
RUNNING_ON_VERCEL = _is_running_on_vercel()
RUNTIME_BASE_DIR = os.getenv(
    "RUNTIME_BASE_DIR",
    os.path.join("/tmp", "bio-agent-runtime") if RUNNING_ON_VERCEL else BASE_DIR,
)
DB_DIR = os.getenv("DB_DIR", os.path.join(RUNTIME_BASE_DIR, "db_runtime"))
DB_PATH = os.getenv("DB_PATH", os.path.join(DB_DIR, "bio_agent.db"))
CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    os.path.join(RUNTIME_BASE_DIR, "chroma_runtime"),
)
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY", "")
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "")
CHROMA_CLOUD_HOST = os.getenv("CHROMA_CLOUD_HOST", "api.trychroma.com")
CHROMA_CLOUD_PORT = _get_int_env("CHROMA_CLOUD_PORT", 8000)
CHROMA_CLOUD_SSL = _get_bool_env("CHROMA_CLOUD_SSL", True)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
CHROMA_PRODUCT_TELEMETRY_IMPL = os.getenv(
    "CHROMA_PRODUCT_TELEMETRY_IMPL",
    "chroma_telemetry.NoOpProductTelemetryClient",
)
CHROMA_USE_CLOUD = bool(CHROMA_API_KEY and CHROMA_TENANT and CHROMA_DATABASE)

# ── API Runtime ────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = _get_int_env("API_PORT", 8000)
CORS_ALLOW_ORIGINS = _get_csv_env(
    "CORS_ALLOW_ORIGINS",
    ["http://localhost:3000"],
)

# ── Model Provider ─────────────────────────────────────────────────────
LLM_API_BASE_URL = os.getenv(
    "LLM_API_BASE_URL",
    os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
)
LLM_API_KEY = os.getenv(
    "LLM_API_KEY",
    os.getenv("OPENAI_API_KEY", "ollama"),
)
AGENT_MODEL = os.getenv("AGENT_MODEL", "llama3.2")

EVALUATOR_API_BASE_URL = os.getenv("EVALUATOR_API_BASE_URL", LLM_API_BASE_URL)
EVALUATOR_API_KEY = os.getenv("EVALUATOR_API_KEY", LLM_API_KEY)
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", AGENT_MODEL)
RUNNING_IN_SPACE = bool(os.getenv("SPACE_ID") or os.getenv("SPACE_REPO_NAME"))
ENABLE_EVALUATION = _get_bool_env("ENABLE_EVALUATION", True)

# ── Evaluator Thresholds ──────────────────────────────────────────────
EVAL_ACCEPT_SCORE = _get_int_env("EVAL_ACCEPT_SCORE", 7)
EVAL_FAQ_SCORE = _get_int_env("EVAL_FAQ_SCORE", 9)
MAX_EVAL_RETRIES = _get_int_env("MAX_EVAL_RETRIES", 2)

# ── RAG Settings ──────────────────────────────────────────────────────
RAG_COLLECTION_NAME = "bio"
RAG_CHUNK_SIZE = _get_int_env("RAG_CHUNK_SIZE", 200)
RAG_CHUNK_OVERLAP = _get_int_env("RAG_CHUNK_OVERLAP", 30)
RAG_TOP_K = _get_int_env("RAG_TOP_K", 3)
CHROMA_ANONYMIZED_TELEMETRY = _get_bool_env("CHROMA_ANONYMIZED_TELEMETRY", False)
