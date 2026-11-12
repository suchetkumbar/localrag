import hashlib
import json
from pathlib import Path

import httpx
import pytest

from rag.corpus import (
    Manifest,
    ManifestFile,
    build_manifest,
    expand_includes,
    fetch,
    include_targets,
    prepare,
    read_fingerprint,
    select_docs,
    strip_heading_anchors,
)

CODE = "line1\nline2\nline3\nline4\n"
PAGE = "# Title { #title }\n\nIntro.\n\n{* ../../docs_src/a/t1.py ln[2:3] hl[2] *}\n\nEnd.\n"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_include_targets_resolve_from_mkdocs_project_dir() -> None:
    md = "{* ../../docs_src/a/t1.py *}\ntext\n{* ../../docs_src/b/t2.py hl[1] *}\n"
    assert include_targets(md) == ["docs_src/a/t1.py", "docs_src/b/t2.py"]


def test_include_path_cannot_escape_repo() -> None:
    with pytest.raises(ValueError, match="escapes"):
        include_targets("{* ../../../../etc/passwd *}")


def test_expand_includes_line_range_and_full_file() -> None:
    def read(path: str) -> str:
        assert path == "docs_src/a/t1.py"
        return CODE

    assert "```python\nline2\nline3\n```" in expand_includes(PAGE, read)
    full = expand_includes("{* ../../docs_src/a/t1.py hl[1] *}", read)
    assert full == "```python\nline1\nline2\nline3\nline4\n```"


def test_dict_unpacking_is_not_an_include() -> None:
    md = "new = {**old, 'k': 1}\n"
    assert expand_includes(md, lambda _: "unused") == md


def test_strip_heading_anchors() -> None:
    md = "## Path params { #path-params }\nText {not an anchor}\n"
    assert strip_heading_anchors(md) == "## Path params\nText {not an anchor}\n"


def test_select_docs_excludes_changelog_and_reference() -> None:
    paths = [
        "docs/en/docs/index.md",
        "docs/en/docs/release-notes.md",
        "docs/en/docs/reference/apirouter.md",
        "docs/en/docs/tutorial/body.md",
        "docs/es/docs/index.md",
        "docs/en/docs/img/logo.png",
    ]
    assert select_docs(paths) == ["docs/en/docs/index.md", "docs/en/docs/tutorial/body.md"]


def _github_mock(files: dict[str, bytes], tree_extra: list[str] | None = None) -> httpx.Client:
    def respond(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/commits/v1"):
            return httpx.Response(200, json={"sha": "abc123"})
        if "/git/trees/abc123" in url:
            paths = list(files) + (tree_extra or [])
            tree = [{"path": p, "type": "blob"} for p in paths]
            return httpx.Response(200, json={"truncated": False, "tree": tree})
        prefix = "https://raw.githubusercontent.com/fastapi/fastapi/abc123/"
        if url.startswith(prefix) and url[len(prefix) :] in files:
            return httpx.Response(200, content=files[url[len(prefix) :]])
        return httpx.Response(404, text="not found")

    return httpx.Client(transport=httpx.MockTransport(respond))


def test_build_manifest_follows_includes() -> None:
    files = {
        "docs/en/docs/page.md": PAGE.encode(),
        "docs_src/a/t1.py": CODE.encode(),
        "docs_src/unused.py": b"never included",
    }
    manifest = build_manifest("v1", _github_mock(files))
    assert manifest.commit == "abc123"
    assert [f.path for f in manifest.files] == ["docs/en/docs/page.md", "docs_src/a/t1.py"]
    assert manifest.files[1].sha256 == _sha(CODE.encode())


def test_build_manifest_fails_on_missing_include() -> None:
    files = {"docs/en/docs/page.md": PAGE.encode()}
    with pytest.raises(RuntimeError, match="not in the repository"):
        build_manifest("v1", _github_mock(files))


def _manifest(files: dict[str, bytes]) -> Manifest:
    entries = [
        ManifestFile(
            p, f"https://raw.githubusercontent.com/fastapi/fastapi/abc123/{p}", _sha(b), len(b)
        )
        for p, b in sorted(files.items())
    ]
    return Manifest(repo="fastapi/fastapi", ref="v1", commit="abc123", license="MIT", files=entries)


def test_fetch_verifies_and_skips_valid_files(tmp_path: Path) -> None:
    files = {"docs/en/docs/page.md": PAGE.encode(), "docs_src/a/t1.py": CODE.encode()}
    manifest = _manifest(files)
    client = _github_mock(files)
    assert fetch(manifest, tmp_path, client) == (2, 0)
    assert fetch(manifest, tmp_path, client) == (0, 2)
    (tmp_path / "docs_src/a/t1.py").write_text("tampered", encoding="utf-8")
    assert fetch(manifest, tmp_path, client) == (1, 1)  # re-downloaded


def test_fetch_rejects_checksum_mismatch(tmp_path: Path) -> None:
    manifest = _manifest({"docs/en/docs/page.md": b"expected"})
    client = _github_mock({"docs/en/docs/page.md": b"something else"})
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        fetch(manifest, tmp_path, client)


def test_prepare_is_deterministic(tmp_path: Path) -> None:
    files = {"docs/en/docs/page.md": PAGE.encode(), "docs_src/a/t1.py": CODE.encode()}
    manifest = _manifest(files)
    fetch(manifest, tmp_path / "raw", _github_mock(files))
    fp1 = prepare(manifest, tmp_path / "raw", tmp_path / "out1")
    fp2 = prepare(manifest, tmp_path / "raw", tmp_path / "out2")
    assert fp1 == fp2 == read_fingerprint(tmp_path / "out1")
    text = (tmp_path / "out1" / "page.md").read_text(encoding="utf-8")
    assert text.startswith("# Title\n")
    assert "line2\nline3" in text and "{*" not in text


def test_manifest_roundtrip(tmp_path: Path) -> None:
    m = _manifest({"docs/en/docs/x.md": b"x"})
    m.save(tmp_path / "m.json")
    assert Manifest.load(tmp_path / "m.json") == m
    assert json.loads((tmp_path / "m.json").read_text())["commit"] == "abc123"


def test_prepare_removes_stale_documents(tmp_path: Path) -> None:
    files = {"docs/en/docs/page.md": PAGE.encode(), "docs_src/a/t1.py": CODE.encode()}
    manifest = _manifest(files)
    fetch(manifest, tmp_path / "raw", _github_mock(files))
    out = tmp_path / "out"
    (out / "old").mkdir(parents=True)
    (out / "old" / "removed.md").write_text("from an earlier manifest", encoding="utf-8")
    prepare(manifest, tmp_path / "raw", out)
    assert sorted(p.name for p in out.rglob("*.md")) == ["page.md"]
