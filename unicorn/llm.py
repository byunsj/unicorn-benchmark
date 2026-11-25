"""Shared vLLM engine lifecycle for the generator and the judge.

Both stages want the same things: an engine sized to whatever GPU memory is
actually free, prompts truncated to fit the context window, and a way to fully
release the GPU afterwards. Generation and judging use models too large to hold
at once, so a run loads them one at a time and calls :meth:`VLLMEngine.unload`
in between.
"""

from __future__ import annotations

import gc
import os
import signal
import threading
from typing import Any

# vLLM occasionally wedges inside a blocking C call, where a Python-level timeout
# cannot reach it. Killing the engine process is the only reliable escape.
GENERATE_TIMEOUT_S = 1800

# Submitting an entire dataset in one call can wedge the engine, so cap
# in-flight requests regardless of the caller's batch size.
MAX_INFLIGHT_PROMPTS = 256


def kill_engine_cores() -> None:
    """Kill any vLLM EngineCore process.

    The engine may be a sibling or grandchild rather than a direct child, so
    scan every process instead of walking our own children.
    """
    try:
        import psutil
    except ImportError:
        return
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = proc.name() or ""
            cmdline = " ".join(proc.cmdline() or [])
        except psutil.Error:
            continue
        if (
            "EngineCore" in name
            or "EngineCore" in cmdline
            or ("vllm" in cmdline.lower() and "engine" in cmdline.lower())
        ):
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except OSError:
                pass


class VLLMEngine:
    """A vLLM engine plus the prompt handling both stages need."""

    def __init__(
        self,
        model_name: str,
        *,
        max_model_len: int | None = None,
        enforce_eager: bool = False,
        gpu_memory_utilization: float | None = None,
        tensor_parallel_size: int | None = None,
        memory_cap: float = 0.85,
        multimodal: bool = False,
        timeout_s: int = GENERATE_TIMEOUT_S,
    ):
        import torch
        from vllm import LLM

        self.model_name = model_name
        self.max_model_len = max_model_len
        self.timeout_s = timeout_s

        # Size to free memory rather than total, so a later stage is not fighting
        # allocations an earlier one has not returned yet.
        if gpu_memory_utilization is None:
            try:
                free, total = torch.cuda.mem_get_info(0)
                gpu_memory_utilization = min(memory_cap, 0.95 * free / total)
            except (RuntimeError, AssertionError):
                gpu_memory_utilization = memory_cap

        kwargs: dict[str, Any] = {
            "model": model_name,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "tensor_parallel_size": tensor_parallel_size or torch.cuda.device_count() or 1,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
        }
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if not multimodal:
            # Prefix caching is disabled deliberately: it buys little here and has
            # been the source of engine wedges.
            kwargs["enable_prefix_caching"] = False
            kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        if self.max_model_len is None:
            try:
                self.max_model_len = self.llm.llm_engine.model_config.max_model_len
            except AttributeError:
                self.max_model_len = None

    def chat_prompt(self, messages: list[dict]) -> str:
        """Render messages with the model's chat template, thinking disabled."""
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def truncate(self, prompt: str, reserve_tokens: int) -> str | dict:
        """Fit *prompt* in the context window, leaving room for the response.

        Over-long prompts make vLLM raise and take the engine down with it, which
        would lose the whole run over one pathological example. Returning token
        ids also avoids a second tokenization inside vLLM.
        """
        if self.max_model_len is None:
            return prompt
        budget = self.max_model_len - reserve_tokens
        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) <= budget:
            return prompt
        return {"prompt_token_ids": token_ids[:budget]}

    def generate(self, prompts: list, sampling_params, desc: str | None = None) -> list:
        """Run generation in bounded chunks, killing the engine if it hangs."""
        from tqdm import tqdm

        chunk = MAX_INFLIGHT_PROMPTS
        batches = range(0, len(prompts), chunk)
        outputs = []
        for start in tqdm(batches, desc=desc, disable=desc is None or len(prompts) <= chunk):
            timer = threading.Timer(self.timeout_s, kill_engine_cores)
            timer.daemon = True
            timer.start()
            try:
                outputs.extend(
                    self.llm.generate(prompts[start : start + chunk], sampling_params, use_tqdm=False)
                )
            finally:
                timer.cancel()
        return outputs

    def unload(self) -> None:
        """Release the GPU so the next stage can claim it."""
        import torch

        kill_engine_cores()
        self.llm = None
        self.tokenizer = None

        # gc.collect() can invoke LLM.__del__(), which talks to the now-dead
        # engine and hangs. The memory is already freed with the process, so
        # bound the wait and move on.
        done = threading.Event()
        thread = threading.Thread(target=lambda: (gc.collect(), done.set()), daemon=True)
        thread.start()
        done.wait(timeout=30)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
