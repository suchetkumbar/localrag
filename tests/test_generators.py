import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag.cache import DiskCache
from rag.config import SamplingParams
from rag.generators import (
    CachedGenerator,
    FakeGenerator,
    OllamaGenerator,
    OpenAICompatibleGenerator,
)
from rag.prompts import REFUSAL, format_context, load_prompt
from rag.types import ScoredChunk
from tests.helpers import make_chunks


def _mock_client(requests: list[httpx.Request], payload: dict[str, Any]) -> httpx.Client:
    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=payload)

    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(respond))


def test_ollama_generator_request_and_usage() -> None:
    requests: list[httpx.Request] = []
    client = _mock_client(
        requests, {"message": {"content": "Answer [1]"}, "prompt_eval_count": 120, "eval_count": 7}
    )
    gen = OllamaGenerator(
        model="llama3", num_ctx=4096, sampling=SamplingParams(seed=42), client=client
    )
    out = gen.generate("SYS", "USER")

    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/api/chat"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert body["options"] == {"temperature": 0.0, "seed": 42, "num_predict": 512, "num_ctx": 4096}
    assert (out.text, out.prompt_tokens, out.completion_tokens) == ("Answer [1]", 120, 7)
    assert out.cached is False


def test_openai_compatible_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "secret")
    requests: list[httpx.Request] = []
    client = _mock_client(
        requests,
        {
            "choices": [{"message": {"content": "Hi"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 1},
        },
    )
    gen = OpenAICompatibleGenerator(
        base_url="https://api.example.com/v1/", model="m", api_key_env="TEST_KEY", client=client
    )
    out = gen.generate("s", "u")
    assert str(requests[0].url) == "https://api.example.com/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert json.loads(requests[0].content)["temperature"] == 0.0
    assert (out.text, out.prompt_tokens, out.completion_tokens) == ("Hi", 9, 1)


def test_openai_compatible_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        OpenAICompatibleGenerator(base_url="http://x", model="m", api_key_env="MISSING_KEY")


def test_cached_generator_returns_original_latency(cache: DiskCache) -> None:
    inner = FakeGenerator("hello")
    gen = CachedGenerator(inner, cache)
    first = gen.generate("s", "u")
    second = gen.generate("s", "u")
    gen.generate("s", "different prompt")
    assert len(inner.calls) == 2
    assert (first.cached, second.cached) == (False, True)
    assert second.text == first.text and second.latency_ms == first.latency_ms


def test_cache_key_includes_generator_identity(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "c.sqlite")
    CachedGenerator(FakeGenerator("a"), cache).generate("s", "u")
    out = CachedGenerator(FakeGenerator("b"), cache).generate("s", "u")
    assert (out.text, out.cached) == ("b", False)
    cache.close()


def test_prompt_v1_loads_and_contains_refusal() -> None:
    p = load_prompt("answer_v1")
    assert REFUSAL in p.system
    rendered = p.render_user(question="Q?", context="CTX")
    assert "CTX" in rendered and rendered.endswith("Question: Q?")
    assert len(p.version_hash) == 12


def test_load_prompt_validates_placeholders(tmp_path: Path) -> None:
    (tmp_path / "bad.system.txt").write_text("s", encoding="utf-8")
    (tmp_path / "bad.user.txt").write_text("no placeholders", encoding="utf-8")
    with pytest.raises(ValueError, match="context"):
        load_prompt("bad", directory=tmp_path)


def test_format_context_numbers_and_skips_oversized() -> None:
    small1, huge, small2 = make_chunks(["short one", "x" * 500, "short two"])
    chunks = [ScoredChunk(small1, 3), ScoredChunk(huge, 2), ScoredChunk(small2, 1)]
    text, included = format_context(chunks, max_chars=200)
    assert [sc.chunk for sc in included] == [small1, small2]  # huge skipped, later one kept
    assert "[1] (source: doc.md)\nshort one" in text
    assert "[2] (source: doc.md)\nshort two" in text
    assert len(text) <= 200
