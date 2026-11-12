"""Reproducible corpus: FastAPI's English documentation at a pinned commit.

Three steps, each idempotent:

    python -m rag.corpus build-manifest --ref 0.162.5   # maintainer only; writes the manifest
    python -m rag.corpus fetch                          # download + verify sha256 of every file
    python -m rag.corpus prepare                        # inline code examples -> data/corpus/

All three outputs (manifest in ``data/manifests/``, raw files in ``data/raw/``, prepared
corpus in ``data/corpus/``) are git-ignored local data. Any change upstream, a corrupted
download or a different preparation shows up as a checksum or fingerprint mismatch.
"""

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import httpx

from rag.http import raise_for_status

REPO = "fastapi/fastapi"
DOCS_ROOT = "docs/en/docs/"
# Include paths in the docs are relative to the mkdocs project directory, not to the
# markdown file: "../../docs_src/x.py" from any page means "docs_src/x.py".
INCLUDE_BASE = PurePosixPath("docs/en")
# release-notes.md is a ~740 KB changelog (about half of all English docs): it would
# dominate retrieval and invite trivia questions. reference/ pages are mkdocstrings stubs
# ("::: fastapi.APIRouter") whose content is generated from code, so they are empty here.
# Files named _*.md (e.g. _llm-test.md) are internal fixtures for FastAPI's translation
# tooling, not documentation.
EXCLUDE = (
    re.compile(r"^docs/en/docs/release-notes\.md$"),
    re.compile(r"^docs/en/docs/reference/"),
    re.compile(r"^docs/en/docs/(.*/)?_[^/]*\.md$"),
)

DEFAULT_MANIFEST = Path("data/manifests/fastapi.json")
DEFAULT_RAW_DIR = Path("data/raw/fastapi")
DEFAULT_CORPUS_DIR = Path("data/corpus/fastapi")
FINGERPRINT_FILE = "CORPUS_FINGERPRINT"

# {* ../../docs_src/body/tutorial001.py ln[7:9] hl[2] *}
_INCLUDE = re.compile(r"^\{\*\s+(?P<path>\S+)(?P<opts>.*?)\*\}[ \t]*$", re.MULTILINE)
_LINES = re.compile(r"ln\[(\d+):(\d+)\]")
# "## Path parameters { #path-parameters }" -> "## Path parameters"
_ANCHOR = re.compile(r"[ \t]*\{[ \t]*#[\w-]+[ \t]*\}[ \t]*$", re.MULTILINE)

_HEADERS = {"User-Agent": "rag-eval-corpus-fetch"}


@dataclass(frozen=True, slots=True)
class ManifestFile:
    path: str  # repository path, e.g. docs/en/docs/tutorial/body.md
    url: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class Manifest:
    repo: str
    ref: str
    commit: str
    license: str
    files: list[ManifestFile]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        files = [ManifestFile(**f) for f in data.pop("files")]
        return cls(files=files, **data)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def include_targets(markdown: str) -> list[str]:
    """Repository paths of the code files a markdown file includes."""
    return sorted({_resolve(INCLUDE_BASE, m.group("path")) for m in _INCLUDE.finditer(markdown)})


def _resolve(base: PurePosixPath, relative: str) -> str:
    parts: list[str] = list(base.parts)
    for part in PurePosixPath(relative).parts:
        if part == "..":
            if not parts:
                raise ValueError(f"include path escapes the repository: {relative}")
            parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


def expand_includes(markdown: str, read_source: Callable[[str], str]) -> str:
    """Replace each ``{* path ln[a:b] hl[...] *}`` with a fenced code block.

    ``ln[a:b]`` selects lines a..b (1-based, inclusive). ``hl[...]`` only highlights lines
    in the rendered site, so it is ignored.
    """

    def replace(m: re.Match[str]) -> str:
        code = read_source(_resolve(INCLUDE_BASE, m.group("path")))
        lines = _LINES.search(m.group("opts"))
        if lines:
            first, last = int(lines.group(1)), int(lines.group(2))
            code = "\n".join(code.split("\n")[first - 1 : last])
        lang = "python" if m.group("path").endswith(".py") else ""
        return f"```{lang}\n{code.rstrip()}\n```"

    return _INCLUDE.sub(replace, markdown)


def strip_heading_anchors(markdown: str) -> str:
    return _ANCHOR.sub("", markdown)


class SourceReader:
    """Reads verified raw files by repository path."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir

    def __call__(self, repo_path: str) -> str:
        text = (self.raw_dir / repo_path).read_text(encoding="utf-8")
        return text.replace("\r\n", "\n")


# --------------------------------------------------------------------------- steps


def _raw_url(commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}"


def select_docs(tree_paths: Iterable[str]) -> list[str]:
    return sorted(
        p
        for p in tree_paths
        if p.startswith(DOCS_ROOT) and p.endswith(".md") and not any(e.match(p) for e in EXCLUDE)
    )


def build_manifest(ref: str, client: httpx.Client) -> Manifest:
    """Resolve ``ref`` to a commit, select docs, follow their includes, checksum it all."""
    commit_resp = client.get(f"https://api.github.com/repos/{REPO}/commits/{ref}")
    raise_for_status(commit_resp)
    commit = str(commit_resp.json()["sha"])
    tree_resp = client.get(f"https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1")
    raise_for_status(tree_resp)
    tree = tree_resp.json()
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned a truncated tree; cannot build a complete manifest")
    blobs = {t["path"] for t in tree["tree"] if t["type"] == "blob"}

    files: dict[str, ManifestFile] = {}

    def add(path: str) -> bytes:
        url = _raw_url(commit, path)
        resp = client.get(url)
        raise_for_status(resp)
        files[path] = ManifestFile(path, url, sha256_bytes(resp.content), len(resp.content))
        return resp.content

    for md in select_docs(blobs):
        content = add(md).decode("utf-8")
        for target in include_targets(content):
            if target not in blobs:
                raise RuntimeError(f"{md} includes {target}, which is not in the repository")
            if target not in files:
                add(target)
    return Manifest(
        repo=REPO,
        ref=ref,
        commit=commit,
        license="MIT (https://github.com/fastapi/fastapi/blob/master/LICENSE)",
        files=sorted(files.values(), key=lambda f: f.path),
    )


def fetch(manifest: Manifest, raw_dir: Path, client: httpx.Client) -> tuple[int, int]:
    """Download files that are missing or fail their checksum. Returns (downloaded, kept)."""
    downloaded = kept = 0
    for f in manifest.files:
        target = raw_dir / f.path
        if target.exists() and sha256_bytes(target.read_bytes()) == f.sha256:
            kept += 1
            continue
        resp = client.get(f.url)
        raise_for_status(resp)
        actual = sha256_bytes(resp.content)
        if actual != f.sha256:
            raise RuntimeError(f"checksum mismatch for {f.path}: expected {f.sha256}, got {actual}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resp.content)
        downloaded += 1
    return downloaded, kept


def prepare(manifest: Manifest, raw_dir: Path, out_dir: Path) -> str:
    """Write the prepared corpus and return its fingerprint.

    The fingerprint is a sha256 over (relative path, prepared text) of every document. The
    golden dataset records it, so evaluating against a different corpus is detected.
    """
    reader = SourceReader(raw_dir)
    digest = hashlib.sha256()
    docs = [
        f.path for f in manifest.files if f.path.startswith(DOCS_ROOT) and f.path.endswith(".md")
    ]
    for repo_path in sorted(docs):
        text = strip_heading_anchors(expand_includes(reader(repo_path), reader))
        rel = repo_path[len(DOCS_ROOT) :]
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        digest.update(rel.encode() + b"\x00" + text.encode("utf-8") + b"\x00")
    # Remove documents left over from an earlier manifest so the corpus matches exactly.
    expected = {out_dir / p[len(DOCS_ROOT) :] for p in docs}
    for stale in [p for p in out_dir.rglob("*.md") if p not in expected]:
        stale.unlink()
    fingerprint = digest.hexdigest()
    (out_dir / FINGERPRINT_FILE).write_text(fingerprint + "\n", encoding="utf-8")
    return fingerprint


def read_fingerprint(corpus_dir: Path) -> str | None:
    path = corpus_dir / FINGERPRINT_FILE
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.corpus", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-manifest", help="pin a git ref and checksum every needed file")
    b.add_argument("--ref", required=True, help="tag, branch or commit of fastapi/fastapi")
    sub.add_parser("fetch", help="download and verify raw files listed in the manifest")
    sub.add_parser("prepare", help="build data/corpus/fastapi from verified raw files")
    for p in sub.choices.values():
        p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
        p.add_argument("--out", type=Path, default=DEFAULT_CORPUS_DIR)
    args = parser.parse_args(argv)

    with httpx.Client(headers=_HEADERS, timeout=60, follow_redirects=True) as client:
        if args.command == "build-manifest":
            manifest = build_manifest(args.ref, client)
            manifest.save(args.manifest)
            total = sum(f.bytes for f in manifest.files)
            print(
                f"{args.manifest}: {len(manifest.files)} files, {total} bytes @ {manifest.commit}"
            )
        elif args.command == "fetch":
            manifest = Manifest.load(args.manifest)
            downloaded, kept = fetch(manifest, args.raw_dir, client)
            print(f"verified {len(manifest.files)} files ({downloaded} downloaded, {kept} cached)")
        else:
            manifest = Manifest.load(args.manifest)
            fingerprint = prepare(manifest, args.raw_dir, args.out)
            print(f"{args.out}: corpus fingerprint {fingerprint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
