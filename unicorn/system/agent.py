"""The research agent that drives the system tier.

The agent talks to an OpenAI-compatible Responses API — in practice a locally
served open-weights model — and decides for itself how many times to search and
what to search for. That is the whole point of this tier: the compressor is no
longer handed a fixed candidate pool, it has to survive queries the agent
invents, and its output feeds back into the agent's next move.

Start the server before running:

    vllm serve openai/gpt-oss-20b --port 8000
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_BASE_URL = "http://localhost:8000/v1"

# Answer format is prescribed so the grader can extract a final answer and a
# confidence without re-reading the whole trajectory.
QUERY_TEMPLATE = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()


@dataclass
class Trajectory:
    query_id: str
    query: str
    answer: str | None
    status: str
    search_calls: int
    search_failures: int
    shown_docids: list[str]
    messages: list = field(default_factory=list)
    error: str | None = None


class Agent:
    def __init__(
        self,
        tool,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "EMPTY",
        max_iterations: int = 100,
        max_output_tokens: int = 10000,
        reasoning_effort: str = "medium",
        request_timeout: float = 600.0,
    ):
        import openai

        self.tool = tool
        self.model = model
        self.max_iterations = max_iterations
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)

    def run(self, query_id: str, question: str) -> Trajectory:
        """Let the agent search until it answers, or until it runs out of turns."""
        import openai

        self.tool.stats.calls = 0
        self.tool.stats.failures = 0
        self.tool.stats.returned_docids = set()

        messages: list = [{"role": "user", "content": QUERY_TEMPLATE.format(question=question)}]
        request = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "tools": [self.tool.definition()],
            "truncation": "auto",
            "reasoning": {"effort": self.reasoning_effort},
        }

        for _ in range(self.max_iterations):
            try:
                response = self.client.responses.create(**request, input=messages)
            except openai.BadRequestError as exc:
                # Almost always a context overflow: the trajectory grew past the
                # window. That is a real outcome for a compressor under test, not
                # an infrastructure error, so record it rather than retrying.
                return self._finish(query_id, question, messages, "incomplete", str(exc))
            except openai.APIError as exc:
                return self._finish(query_id, question, messages, "error", str(exc))

            output = [_normalize(item) for item in response.model_dump()["output"]]
            messages.extend(output)

            calls = [item for item in output if item.get("type") == "function_call"]
            if not calls:
                return self._finish(query_id, question, messages, "completed", None)

            for call in calls:
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": self._execute(call),
                    }
                )

        return self._finish(query_id, question, messages, "max_iterations", None)

    def _execute(self, call: dict) -> str:
        import json

        try:
            arguments = json.loads(call.get("arguments") or "{}")
            return self.tool(arguments["user_query"])
        except Exception as exc:  # noqa: BLE001 - report tool failures to the agent, never abort the run
            self.tool.stats.failures += 1
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def _finish(self, query_id, question, messages, status, error) -> Trajectory:
        return Trajectory(
            query_id=query_id,
            query=question,
            answer=_final_text(messages),
            status=status,
            search_calls=self.tool.stats.calls,
            search_failures=self.tool.stats.failures,
            shown_docids=sorted(self.tool.stats.returned_docids),
            messages=messages,
            error=error,
        )


def _normalize(item: dict) -> dict:
    """Rewrite a tool call into the one shape the server accepts back as input.

    The same logical call comes back as either ``function_call`` or ``mcp_call``
    depending on how the model phrased it, but only the former is valid on input.
    Echoing an ``mcp_call`` back is rejected and ends the trajectory, so convert
    it and drop the fields that are output-only.
    """
    if item.get("type") != "mcp_call":
        return item
    return {
        "type": "function_call",
        "call_id": item["id"],
        "name": item["name"],
        "arguments": item.get("arguments") or "{}",
    }


def _final_text(messages: list) -> str | None:
    """Pull the agent's last assistant message out of the trajectory."""
    for item in reversed(messages):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        parts = [c.get("text", "") for c in item.get("content") or [] if c.get("type") == "output_text"]
        if parts:
            return "".join(parts).strip()
    return None
