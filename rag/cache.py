"""On-disk cache for model calls (embeddings, reranker scores, LLM generations).

Why: an eval run makes thousands of model calls. Caching them by their exact inputs
makes re-runs nearly free, and makes results reproducible: the same config over the same
dataset yields byte-identical outputs even though the underlying model is a service.

Storage is a single SQLite file (stdlib, no server, safe for concurrent readers).
"""

import hashlib
import io
import json
import sqlite3
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np

from rag.config import CacheMode

# Bump to invalidate every entry, e.g. after changing how values are serialised.
CACHE_SCHEMA_VERSION = 1

T = TypeVar("T")


class CacheMissError(RuntimeError):
    """Raised in ``readonly`` mode when a value is not in the cache."""


class Codec(Protocol[T]):
    def encode(self, value: T) -> bytes: ...

    def decode(self, data: bytes) -> T: ...


class JsonCodec:
    def encode(self, value: Any) -> bytes:
        return json.dumps(value, ensure_ascii=False).encode("utf-8")

    def decode(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"))


class ArrayCodec:
    """numpy's .npy format: keeps dtype and shape, and never unpickles."""

    def encode(self, value: np.ndarray[Any, Any]) -> bytes:
        buf = io.BytesIO()
        np.save(buf, value, allow_pickle=False)
        return buf.getvalue()

    def decode(self, data: bytes) -> np.ndarray[Any, Any]:
        result: np.ndarray[Any, Any] = np.load(io.BytesIO(data), allow_pickle=False)
        return result


JSON_CODEC = JsonCodec()
ARRAY_CODEC = ArrayCodec()


def make_key(kind: str, payload: Mapping[str, Any]) -> str:
    """sha256 over canonical JSON. Key order and whitespace never change the key."""
    canonical = json.dumps(
        {"v": CACHE_SCHEMA_VERSION, "kind": kind, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, path: Path | str, mode: CacheMode = "rw") -> None:
        self.mode: CacheMode = mode
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if mode != "off":
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB NOT NULL)"
            )
            self._conn.commit()

    @classmethod
    def disabled(cls) -> "DiskCache":
        return cls(":unused:", mode="off")

    def get(self, key: str) -> bytes | None:
        if self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return None if row is None else bytes(row[0])

    def put(self, key: str, value: bytes) -> None:
        if self._conn is None or self.mode != "rw":
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
            )
            self._conn.commit()

    def get_or_compute(
        self,
        kind: str,
        payload: Mapping[str, Any],
        compute: Callable[[], T],
        codec: Codec[T],
    ) -> tuple[T, bool]:
        """Return ``(value, was_cached)``."""
        if self.mode == "off":
            return compute(), False
        key = make_key(kind, payload)
        data = self.get(key)
        if data is not None:
            self.hits += 1
            return codec.decode(data), True
        self.misses += 1
        if self.mode == "readonly":
            raise CacheMissError(f"cache miss in readonly mode for {kind} (key {key[:12]})")
        value = compute()
        self.put(key, codec.encode(value))
        return value, False

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
