"""
Bio Agent — Core Agent
----------------------
Pure retrieval-augmented chat over the profile knowledge base.
The backend fetches relevant Chroma context first, then generates a response
grounded in those facts. No SQLite cache, tool calls, or evaluator loop.
"""

from collections.abc import Iterator

from openai import OpenAI

import rag
from config import AGENT_MODEL, LLM_API_BASE_URL, LLM_API_KEY, RUNNING_IN_SPACE, is_local_base_url


class BioAgent:
    """
    A grounded career assistant that:
    1. Searches the Chroma-backed knowledge base
    2. Injects the retrieved facts into the prompt
    3. Answers directly from that context
    """

    def __init__(self):
        self._client = OpenAI(base_url=LLM_API_BASE_URL, api_key=LLM_API_KEY)

    def _system_prompt(self) -> str:
        return """You are acting as a professional career assistant, representing the person described in the knowledge base. You answer questions on their behalf — about their career, skills, experience, projects, and professional background.

## Rules
- Stay in character at all times.
- Only state facts that come from the provided knowledge base context. Do not fabricate details.
- Answer directly and concretely when the context contains the answer.
- If the context does not contain the answer, say so briefly and honestly.
- Never use placeholders such as `[specific skills]`, `[technologies]`, or similar bracketed filler.
- Be warm, professional, and concise.
"""

    def _normalize_history(self, history: list[dict] | None) -> list[dict]:
        """Sanitize chat history so only valid user/assistant turns are reused."""
        normalized_history = []

        for item in history or []:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue

            content = content.strip()
            if not content:
                continue

            normalized_history.append({"role": role, "content": content})

        return normalized_history

    def _get_factual_context(self, message: str) -> str:
        """Fetch relevant knowledge base context for the user's question."""
        context = rag.search_knowledge_base(message)
        unavailable_prefixes = (
            "No knowledge base documents found.",
            "Knowledge base retrieval is temporarily unavailable.",
            "No relevant information found in the knowledge base.",
        )
        if any(context.startswith(prefix) for prefix in unavailable_prefixes):
            return ""
        return context

    def _build_messages(
        self,
        message: str,
        history: list[dict] | None = None,
        factual_context: str = "",
    ) -> list[dict]:
        """Build the model message list from system instructions, history, and context."""
        messages = [{"role": "system", "content": self._system_prompt()}]

        if factual_context.strip():
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Use this factual context to answer the user's question.\n\n"
                        f"{factual_context}\n\n"
                        "If this context contains the answer, respond with those concrete facts. "
                        "Do not claim the information is unavailable when it is present here."
                    ),
                }
            )

        messages.extend(self._normalize_history(history))
        messages.append({"role": "user", "content": message})
        return messages

    def _complete(self, messages: list[dict]) -> str:
        """Generate a grounded assistant answer without relying on tool calls."""
        response = self._client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    def _complete_stream(self, messages: list[dict]) -> Iterator[dict]:
        """Stream a grounded assistant answer without relying on tool calls."""
        stream = self._client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            stream=True,
        )

        answer_parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta.content:
                answer_parts.append(delta.content)
                yield {"type": "delta", "text": delta.content}

        answer = "".join(answer_parts)
        if not answer.strip():
            answer = self._complete(messages)

        yield {"type": "done", "answer": answer}

    def chat(self, message: str, history: list[dict] | None = None) -> str:
        """Answer a question using Chroma-retrieved factual context."""
        history = history or []

        if RUNNING_IN_SPACE and is_local_base_url(LLM_API_BASE_URL):
            return (
                "This Space is still configured for local Ollama at "
                "`http://localhost:11434/v1`, which does not exist on Hugging Face Spaces. "
                "Set `LLM_API_BASE_URL`, `LLM_API_KEY`, and `AGENT_MODEL` in the Space settings, "
                "then restart the Space."
            )

        context = self._get_factual_context(message)
        messages = self._build_messages(
            message=message,
            history=history,
            factual_context=context,
        )

        try:
            return self._complete(messages)
        except Exception as exc:
            return (
                "The assistant could not reach its configured language model provider. "
                "Check `LLM_API_BASE_URL`, `LLM_API_KEY`, and `AGENT_MODEL`. "
                f"Runtime error: {type(exc).__name__}: {exc}"
            )

    def stream_chat(self, message: str, history: list[dict] | None = None) -> Iterator[str]:
        """Stream an answer using Chroma-retrieved factual context."""
        history = history or []

        if RUNNING_IN_SPACE and is_local_base_url(LLM_API_BASE_URL):
            yield (
                "This Space is still configured for local Ollama at "
                "`http://localhost:11434/v1`, which does not exist on Hugging Face Spaces. "
                "Set `LLM_API_BASE_URL`, `LLM_API_KEY`, and `AGENT_MODEL` in the Space settings, "
                "then restart the Space."
            )
            return

        context = self._get_factual_context(message)
        messages = self._build_messages(
            message=message,
            history=history,
            factual_context=context,
        )

        try:
            answer = ""
            for event in self._complete_stream(messages):
                if event["type"] == "delta":
                    answer += event["text"]
                    yield event["text"]
                elif not answer and event["answer"]:
                    yield event["answer"]
        except Exception as exc:
            yield (
                "The assistant could not reach its configured language model provider. "
                "Check `LLM_API_BASE_URL`, `LLM_API_KEY`, and `AGENT_MODEL`. "
                f"Runtime error: {type(exc).__name__}: {exc}"
            )
