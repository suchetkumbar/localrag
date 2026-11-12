from pathlib import Path

import numpy as np
import pytest

from rag.cache import ARRAY_CODEC, JSON_CODEC, CacheMissError, DiskCache, make_key


def test_key_ignores_dict_order() -> None:
    assert make_key("k", {"a": 1, "b": 2}) == make_key("k", {"b": 2, "a": 1})


@pytest.mark.parametrize(
    "other",
    [
        ("k2", {"model": "m", "text": "t"}),
        ("k", {"model": "m2", "text": "t"}),
        ("k", {"model": "m", "text": "t "}),
        ("k", {"model": "m", "text": "t", "temperature": 0}),
    ],
)
def test_key_changes_with_any_field(other: tuple[str, dict[str, object]]) -> None:
    assert make_key("k", {"model": "m", "text": "t"}) != make_key(*other)


def test_rw_computes_once_then_hits(cache: DiskCache) -> None:
    calls = []

    def compute() -> dict[str, int]:
        calls.append(1)
        return {"x": 1}

    v1, cached1 = cache.get_or_compute("k", {"p": 1}, compute, JSON_CODEC)
    v2, cached2 = cache.get_or_compute("k", {"p": 1}, compute, JSON_CODEC)
    assert v1 == v2 == {"x": 1}
    assert (cached1, cached2) == (False, True)
    assert len(calls) == 1
    assert (cache.hits, cache.misses) == (1, 1)


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    first = DiskCache(path)
    first.get_or_compute("k", {"p": 1}, lambda: "value", JSON_CODEC)
    first.close()
    second = DiskCache(path, mode="readonly")
    value, cached = second.get_or_compute("k", {"p": 1}, lambda: "other", JSON_CODEC)
    assert (value, cached) == ("value", True)
    second.close()


def test_readonly_miss_raises(tmp_path: Path) -> None:
    c = DiskCache(tmp_path / "c.sqlite", mode="readonly")
    with pytest.raises(CacheMissError):
        c.get_or_compute("k", {"p": 1}, lambda: 1, JSON_CODEC)
    c.close()


def test_off_mode_always_computes_and_stores_nothing() -> None:
    c = DiskCache.disabled()
    calls = []
    for _ in range(2):
        c.get_or_compute("k", {}, lambda: calls.append(1), JSON_CODEC)
    assert len(calls) == 2
    assert c.get(make_key("k", {})) is None


def test_array_codec_roundtrip_keeps_dtype_and_shape() -> None:
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = ARRAY_CODEC.decode(ARRAY_CODEC.encode(arr))
    assert out.dtype == np.float32
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, arr)
