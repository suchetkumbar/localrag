import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag.cache import DiskCache
from rag.config import PipelineConfig
from rag.corpus import FINGERPRINT_FILE
from rag.eval.evaluate import EvalConfig
from rag.golden.schema import GoldenItem, Provenance, Review, write_jsonl
from rag.prompts import REFUSAL
from rag.types import Document
from tests.helpers import golden_item

SAMPLE_MD = """# Security

FastAPI provides several tools to help you deal with security.

## OAuth2

OAuth2 is a specification that defines several ways to handle authentication.

```python
# This is a comment, not a heading
def get_user(): ...
```

### Scopes

Scopes let a client request only specific permissions from the user.

## HTTP Basic Auth

The simplest authentication sends a username and password in a header.
"""


@pytest.fixture
def sample_doc() -> Document:
    return Document(doc_id="tutorial/security.md", text=SAMPLE_MD, title="Security")


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[DiskCache]:
    c = DiskCache(tmp_path / "cache.sqlite", mode="rw")
    yield c
    c.close()


# Shared by the experiment and gate tests: a tiny corpus, a verified 3-item dataset,
# fake models and a config factory.
SECTION = (
    "## Ports\n\nThe development server listens on port 8000 by default. "
    "You can change it with the --port option.\n\n"
    "## Validation\n\nInvalid request bodies return status code 422 with details of every error.\n"
)


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Any]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = "# Server\n\n" + SECTION
    (corpus / "ports.md").write_text(doc, encoding="utf-8")
    (corpus / FINGERPRINT_FILE).write_text("fp\n", encoding="utf-8")
    items = [
        golden_item("q1", doc, "The development server listens on port 8000 by default."),
        golden_item(
            "q2", doc, "Invalid request bodies return status code 422 with details of every error."
        ),
        GoldenItem(
            id="u1",
            question="Does FastAPI ship a GraphQL server?",
            reference_answer=REFUSAL,
            question_type="unanswerable",
            human_verified=True,
            review=Review(status="accepted"),
            provenance=Provenance(drafted_by="t", draft_prompt="t"),
            corpus_fingerprint="fp",
        ),
    ]
    dataset = tmp_path / "v1.jsonl"
    write_jsonl(dataset, items)

    def config(name: str, retriever: str) -> PipelineConfig:
        return PipelineConfig.model_validate(
            {
                "name": name,
                "corpus": {"path": str(corpus)},
                "chunker": {"kind": "heading", "max_size": 300, "overlap": 50},
                "embedder": {"kind": "fake"},
                "retrieval": {"retriever": {"kind": retriever}, "candidate_k": 10, "top_k": 2},
                "generator": {"kind": "fake", "response": "Port 8000 [1]."},
                "cache": {"path": str(tmp_path / "cache.sqlite")},
            }
        )

    judge_json = json.dumps(
        {"reason": "ok", "label": "correct", "claims": [{"claim": "port 8000", "supported": True}]}
    )
    eval_cfg = EvalConfig.model_validate({"judge": {"kind": "fake", "response": judge_json}})
    return {"tmp": tmp_path, "dataset": dataset, "config": config, "eval": eval_cfg, "doc": doc}
