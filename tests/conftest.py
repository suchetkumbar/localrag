from collections.abc import Iterator
from pathlib import Path

import pytest

from rag.cache import DiskCache
from rag.types import Document

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
