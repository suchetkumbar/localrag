"""LLM generators: Ollama (local, default), any OpenAI-compatible API, and a fake.

All three talk plain HTTP through httpx, with no vendor SDK. "OpenAI-compatible" covers
free hosted tiers such as Groq, OpenRouter and Gemini as well as Ollama's own /v1
endpoint, so one class gives a provider-agnostic hosted option with zero extra deps.
"""

import os
import time
from collections.abc import Mapping
from typing import Any

import httpx

from rag.cache import JSON_CODEC, DiskCache
from rag.config import SamplingParams
from rag.http import raise_for_status
from rag.interfaces import Generator
from rag.types import Generation


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class OllamaGenerator:
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        num_ctx: int = 8192,
        sampling: SamplingParams | None = None,
        timeout_s: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.sampling = sampling or SamplingParams()
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": "ollama",
            "model": self.model,
            "num_ctx": self.num_ctx,
            **self.sampling.model_dump(),
        }

    def generate(self, system: str, user: str) -> Generation:
        body = {
            "model": self.model,
            "messages": _messages(system, user),
            "stream": False,
            "options": {
                "temperature": self.sampling.temperature,
                "seed": self.sampling.seed,
                "num_predict": self.sampling.max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        t0 = time.perf_counter()
        resp = self._client.post("/api/chat", json=body)
        raise_for_status(resp)
        data = resp.json()
        return Generation(
            text=data["message"]["content"],
            model=self.model,
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class OpenAICompatibleGenerator:
    """POST {base_url}/chat/completions with a bearer key read from an env var."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str,
        sampling: SamplingParams | None = None,
        timeout_s: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(f"environment variable {api_key_env} is not set (see .env.example)")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sampling = sampling or SamplingParams()
        self._client = client or httpx.Client(timeout=timeout_s)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    @property
    def identity(self) -> Mapping[str, Any]:
        # base_url is part of identity: the "same" model name can differ across providers.
        return {"provider": self.base_url, "model": self.model, **self.sampling.model_dump()}

    def generate(self, system: str, user: str) -> Generation:
        body = {
            "model": self.model,
            "messages": _messages(system, user),
            "temperature": self.sampling.temperature,
            "seed": self.sampling.seed,
            "max_tokens": self.sampling.max_tokens,
        }
        t0 = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}/chat/completions", json=body, headers=self._headers
        )
        raise_for_status(resp)
        data = resp.json()
        usage = data.get("usage") or {}
        return Generation(
            text=data["choices"][0]["message"]["content"] or "",
            model=self.model,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class FakeGenerator:
    """Returns a fixed response. For unit tests of the pipeline wiring only."""

    def __init__(self, response: str = "This is a fake answer [1].") -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"provider": "fake", "response": self.response}

    def generate(self, system: str, user: str) -> Generation:
        self.calls.append((system, user))
        return Generation(
            text=self.response,
            model="fake",
            prompt_tokens=len((system + user).split()),
            completion_tokens=len(self.response.split()),
            latency_ms=0.0,
        )


class CachedGenerator:
    """Caches generations keyed by (generator identity, system prompt, user prompt).

    A cache hit returns the *original* latency and token counts with ``cached=True``,
    so operational metrics from a cached re-run still describe the real model call.
    """

    def __init__(self, inner: Generator, cache: DiskCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def identity(self) -> Mapping[str, Any]:
        return self.inner.identity

    def generate(self, system: str, user: str) -> Generation:
        def compute() -> dict[str, Any]:
            g = self.inner.generate(system, user)
            return {
                "text": g.text,
                "model": g.model,
                "prompt_tokens": g.prompt_tokens,
                "completion_tokens": g.completion_tokens,
                "latency_ms": g.latency_ms,
            }

        value, was_cached = self.cache.get_or_compute(
            "generation",
            {"generator": dict(self.inner.identity), "system": system, "user": user},
            compute,
            JSON_CODEC,
        )
        return Generation(**value, cached=was_cached)
