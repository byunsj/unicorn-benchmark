"""Pointwise LLM judge scoring answers against the gold answer.

The judge is asked a single Yes/No question and scored from the *logits* of that
first token rather than the sampled string. Two reasons: the score is continuous,
so a confidently-correct answer is distinguishable from a borderline one, and it
cannot be derailed by the model wrapping its verdict in a sentence.

This reproduces Bergen's ``LLMeval`` with ``use_logits=True``, so numbers stay
comparable with published results.
"""

from __future__ import annotations

import math

from .llm import VLLMEngine

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"

MODEL_ALIASES = {"Qwen3.5-35B": "Qwen/Qwen3.5-35B-A3B"}

# The verdict vocabulary, and the score each verdict contributes.
OPTIONS = {"Yes": 1.0, "No": 0.0}

SYSTEM_PROMPT = "You are an evaluation tool. Answer with one of Yes, No."

USER_TEMPLATE = (
    "Here is a question, a golden answer and an AI-generated answer. "
    "Can you judge whether the AI-generated answer is correct according "
    "to the question and golden answer, simply answer with one of Yes, No.\n"
    "Question: {question}\nGolden answer: {answer}\nGenerated answer: {prediction}"
)


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


class Judge:
    """Score generated answers in [0, 1] from the judge's Yes/No logits."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        max_model_len: int = 4096,
        **engine_kwargs,
    ):
        self.model_name = resolve_model(model_name)
        self.max_model_len = max_model_len
        # enforce_eager avoids a long graph capture for a model used to emit one
        # token per prompt.
        self.engine = VLLMEngine(
            self.model_name,
            max_model_len=max_model_len,
            enforce_eager=True,
            memory_cap=0.92,
            **engine_kwargs,
        )

        options = sorted(OPTIONS)
        self.option_token_ids = [
            self.engine.tokenizer.encode(o, add_special_tokens=False)[0] for o in options
        ]
        self.option_values = [OPTIONS[o] for o in options]

    def _prompt(self, question: str, answer: str, prediction: str) -> str | dict:
        content = USER_TEMPLATE.format(question=question, answer=answer, prediction=prediction)
        tokenizer = self.engine.tokenizer
        if getattr(tokenizer, "chat_template", None):
            prompt = self.engine.chat_prompt(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ]
            )
        else:
            prompt = f"{SYSTEM_PROMPT}\n{content}"
        return self.engine.truncate(prompt, reserve_tokens=1)

    def _score(self, logprobs: dict) -> float:
        """Softmax over the Yes/No logprobs, then take the expected value."""
        pairs = [
            (logprobs[tid].logprob if tid in logprobs else float("-inf"), value)
            for tid, value in zip(self.option_token_ids, self.option_values)
        ]
        ceiling = max(lp for lp, _ in pairs)
        weights = [math.exp(lp - ceiling) for lp, _ in pairs]
        total = sum(weights)
        return sum((w / total) * value for w, (_, value) in zip(weights, pairs))

    def score(self, questions: list[str], answers: list[str], predictions: list[str]) -> list[float]:
        from vllm import SamplingParams

        if not (len(questions) == len(answers) == len(predictions)):
            raise ValueError("questions, answers, and predictions must be the same length")

        prompts = [self._prompt(q, a, p) for q, a, p in zip(questions, answers, predictions)]
        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=len(self.option_token_ids),
            allowed_token_ids=self.option_token_ids,
        )
        outputs = self.engine.generate(prompts, params, desc="Judge")
        return [self._score(out.outputs[0].logprobs[0]) for out in outputs]

    def metric_key(self) -> str:
        return f"LLMeval_{self.model_name.replace('/', '_')}_logits"

    def unload(self) -> None:
        self.engine.unload()
