"""
Bio Agent — Core Agent
-----------------------
Orchestrates the agent loop, tool dispatch, evaluation, and reflection.
This is the central class that ties everything together.
"""

from collections.abc import Iterator
import inspect
import json

from openai import OpenAI

import database
import rag
import evaluator
from tools import TOOLS_LIST, TOOLS_MAP
from config import (
    LLM_API_BASE_URL,
    LLM_API_KEY,
    AGENT_MODEL,
    EVAL_ACCEPT_SCORE,
    EVAL_FAQ_SCORE,
    MAX_EVAL_RETRIES,
    ENABLE_EVALUATION,
    RUNNING_IN_SPACE,
    is_local_base_url,
)


class BioAgent:
    """
    A self-improving career assistant that:
    1. Checks FAQ cache before doing expensive LLM + RAG calls
    2. Searches a ChromaDB knowledge base for factual answers
    3. Evaluates its own responses via a separate LLM judge
    4. Refines responses that score below threshold (reflection)
    5. Promotes excellent answers to FAQ for future reuse
    """

    def __init__(self):
        self._client = OpenAI(base_url=LLM_API_BASE_URL, api_key=LLM_API_KEY)

        # Initialise database tables
        database.init_db()

        # Ingest knowledge base (idempotent — skips if already done)
        chunk_count = rag.ingest_knowledge()
        print(f"[BioAgent] Knowledge base ready — {chunk_count} chunks indexed.")

    # ── System Prompt ─────────────────────────────────────────────────

    def _system_prompt(self) -> str:
        return """You are acting as a professional career assistant, representing the person described in the knowledge base. You answer questions on their behalf — about their career, skills, experience, projects, and professional background.

## Your Workflow
1. **ALWAYS call `lookup_faq` first** with the user's question. If a cached answer exists, use it directly.
2. If no FAQ match, call `search_knowledge_base` with a relevant query to retrieve factual context.
3. Use the retrieved context to craft an accurate, professional response.
4. If a user shares their email or wants to connect, call `record_contact` to save their details.

## Rules
- Stay in character at all times — you ARE this professional person.
- Only state facts that come from the knowledge base or FAQ. Do not fabricate details.
- Be warm, professional, and engaging — as if speaking to a potential employer or collaborator.
- If you cannot find an answer in the knowledge base, say so honestly rather than guessing.
- Gently steer conversations toward professional topics and encourage users to get in touch.
"""

    # ── Tool Dispatch ─────────────────────────────────────────────────

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

    def _build_messages(self, message: str, history: list[dict] | None = None) -> list[dict]:
        """Build the model message list from the system prompt, history, and new input."""
        return (
            [{"role": "system", "content": self._system_prompt()}]
            + self._normalize_history(history)
            + [{"role": "user", "content": message}]
        )

    def _get_tool_call_parts(self, tool_call) -> tuple[str, str, str]:
        """Read a tool call from either an SDK object or a plain dict."""
        if isinstance(tool_call, dict):
            function = tool_call.get("function", {})
            return (
                tool_call.get("id", ""),
                function.get("name", ""),
                function.get("arguments", "{}"),
            )

        return (
            tool_call.id,
            tool_call.function.name,
            tool_call.function.arguments,
        )

    def _handle_tool_calls(self, tool_calls) -> tuple[list[dict], str]:
        """
        Execute tool calls and return (results_messages, last_context).
        Captures RAG context for the evaluator.
        """
        results = []
        context = ""

        for tool_call in tool_calls:
            tool_call_id, name, raw_arguments = self._get_tool_call_parts(tool_call)
            args = json.loads(raw_arguments or "{}")

            print(f"  [Tool] {name}({args})")

            func = TOOLS_MAP.get(name)
            if func:
                # Filter args to only parameters the function accepts.
                # Small LLMs sometimes hallucinate extra keys.
                sig = inspect.signature(func)
                valid_params = set(sig.parameters.keys())
                filtered_args = {k: v for k, v in args.items() if k in valid_params}

                if filtered_args != args:
                    dropped = set(args.keys()) - valid_params
                    print(f"  [Warning] Dropped unexpected args: {dropped}")

                result = func(**filtered_args)
                # Capture RAG context for evaluation
                if name == "search_knowledge_base":
                    context = result
            else:
                result = json.dumps({"error": f"Unknown tool: {name}"})

            results.append({
                "role": "tool",
                "content": result if isinstance(result, str) else json.dumps(result),
                "tool_call_id": tool_call_id,
            })

        return results, context

    def _tool_message_from_calls(self, tool_calls: list[dict]) -> dict:
        """Convert plain tool call dicts into a chat message payload."""
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls,
        }

    # ── Agent Loop ────────────────────────────────────────────────────

    def _run_agent_loop(self, messages: list[dict]) -> tuple[str, str]:
        """
        Run the while-not-done agent loop.
        Returns (agent_answer, rag_context_used).
        """
        context = ""

        while True:
            response = self._client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                tools=TOOLS_LIST,
            )

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                message = choice.message
                tool_calls = message.tool_calls
                tool_results, tool_context = self._handle_tool_calls(tool_calls)

                if tool_context:
                    context = tool_context

                messages.append(self._tool_message_from_calls([
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ]))
                messages.extend(tool_results)
            else:
                # LLM produced a final text response
                return choice.message.content or "", context

    def _run_agent_loop_stream(self, messages: list[dict]) -> Iterator[dict]:
        """
        Stream the final assistant text while still supporting the internal tool loop.
        Tool-calling turns are handled server-side; only final text deltas are yielded.
        """
        context = ""

        while True:
            stream = self._client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                tools=TOOLS_LIST,
                stream=True,
            )

            finish_reason = ""
            answer_parts: list[str] = []
            streamed_tool_calls: dict[int, dict] = {}

            for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta

                if delta.content:
                    answer_parts.append(delta.content)
                    yield {"type": "delta", "text": delta.content}

                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        call_state = streamed_tool_calls.setdefault(
                            tool_call.index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": "",
                                },
                            },
                        )

                        if tool_call.id:
                            call_state["id"] = tool_call.id
                        if tool_call.type:
                            call_state["type"] = tool_call.type
                        if tool_call.function:
                            if tool_call.function.name:
                                call_state["function"]["name"] += tool_call.function.name
                            if tool_call.function.arguments:
                                call_state["function"]["arguments"] += tool_call.function.arguments

            if finish_reason == "tool_calls":
                tool_calls = [
                    streamed_tool_calls[index]
                    for index in sorted(streamed_tool_calls.keys())
                ]
                tool_results, tool_context = self._handle_tool_calls(tool_calls)

                if tool_context:
                    context = tool_context

                messages.append(self._tool_message_from_calls(tool_calls))
                messages.extend(tool_results)
                continue

            yield {
                "type": "done",
                "answer": "".join(answer_parts),
                "context": context,
            }
            return

    # ── Public Chat Interface ─────────────────────────────────────────

    def chat(self, message: str, history: list[dict] | None = None) -> str:
        """
        Main public chat entry point. Handles:
        1. Agent loop (tool calling + response generation)
        2. Evaluation (LLM-as-judge scoring)
        3. Reflection (retry if score < threshold)
        4. Persistence (log conversation, promote to FAQ)
        """
        history = history or []

        if RUNNING_IN_SPACE and is_local_base_url(LLM_API_BASE_URL):
            return (
                "This Space is still configured for local Ollama at "
                "`http://localhost:11434/v1`, which does not exist on Hugging Face Spaces. "
                "Set `LLM_API_BASE_URL`, `LLM_API_KEY`, and `AGENT_MODEL` in the Space settings, "
                "then restart the Space."
            )

        messages = self._build_messages(message=message, history=history)

        answer = ""
        context = ""
        score = 0

        try:
            if not ENABLE_EVALUATION:
                answer, _ = self._run_agent_loop(messages)
                score = EVAL_ACCEPT_SCORE
            else:
                for attempt in range(1 + MAX_EVAL_RETRIES):
                    answer, loop_context = self._run_agent_loop(messages)
                    if loop_context:
                        context = loop_context

                    # Evaluate the response
                    eval_result = evaluator.evaluate_response(
                        user_question=message,
                        agent_answer=answer,
                        context=context,
                    )
                    score = eval_result["score"]
                    feedback = eval_result["feedback"]

                    print(f"  [Eval] Attempt {attempt + 1} — Score: {score}/10 — {feedback}")

                    if score >= EVAL_ACCEPT_SCORE:
                        break  # Good enough — accept

                    # Reflection: feed evaluator feedback back and retry
                    messages.append({"role": "assistant", "content": answer})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your previous response scored {score}/10. "
                            f"Evaluator feedback: {feedback}\n\n"
                            "Please improve your response based on this feedback."
                        ),
                    })
                    print(f"  [Reflection] Retrying with evaluator feedback...")
        except Exception as exc:
            return (
                "The assistant could not reach its configured language model provider. "
                "Check `LLM_API_BASE_URL`, `LLM_API_KEY`, and `AGENT_MODEL`. "
                f"Runtime error: {type(exc).__name__}: {exc}"
            )

        # ── Persist Results ───────────────────────────────────────────

        # Always log the conversation
        database.log_conversation(
            user_question=message,
            agent_answer=answer,
            eval_score=score,
        )

        # Promote excellent answers to FAQ
        if score >= EVAL_FAQ_SCORE:
            database.save_faq(question=message, answer=answer)
            print(f"  [FAQ] Answer promoted to FAQ (score {score})")

        return answer

    def stream_chat(self, message: str, history: list[dict] | None = None) -> Iterator[str]:
        """
        Stream assistant text deltas for the frontend.
        Streaming skips evaluator retries so the UI can render incrementally.
        """
        history = history or []

        if RUNNING_IN_SPACE and is_local_base_url(LLM_API_BASE_URL):
            yield (
                "This Space is still configured for local Ollama at "
                "`http://localhost:11434/v1`, which does not exist on Hugging Face Spaces. "
                "Set `LLM_API_BASE_URL`, `LLM_API_KEY`, and `AGENT_MODEL` in the Space settings, "
                "then restart the Space."
            )
            return

        messages = self._build_messages(message=message, history=history)

        answer = ""
        score = EVAL_ACCEPT_SCORE

        try:
            for event in self._run_agent_loop_stream(messages):
                if event["type"] == "delta":
                    answer += event["text"]
                    yield event["text"]
                elif not answer and event["answer"]:
                    answer = event["answer"]
                    yield answer
        except Exception as exc:
            answer = (
                "The assistant could not reach its configured language model provider. "
                "Check `LLM_API_BASE_URL`, `LLM_API_KEY`, and `AGENT_MODEL`. "
                f"Runtime error: {type(exc).__name__}: {exc}"
            )
            yield answer

        database.log_conversation(
            user_question=message,
            agent_answer=answer,
            eval_score=score,
        )
