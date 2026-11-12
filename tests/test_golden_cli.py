"""End-to-end tests of the build / validate / stats commands on a tiny corpus."""

from pathlib import Path

import pytest

from rag.corpus import FINGERPRINT_FILE
from rag.golden import __main__ as cli
from rag.golden.card import END, START
from rag.golden.schema import Evidence, GoldenItem, Review, read_jsonl, write_jsonl
from tests.test_golden_schema import DOC, make_item


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "d.md").write_text(DOC.text, encoding="utf-8")
    (corpus / FINGERPRINT_FILE).write_text("fp\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(f"# Card\n{START}\n{END}\n", encoding="utf-8")
    monkeypatch.setattr(
        cli, "QUOTAS", {"factual": 2, "numeric": 0, "multi_hop": 0, "unanswerable": 0}
    )
    return tmp_path


def _args(ws: Path, *extra: str) -> list[str]:
    return [*extra, "--drafts", str(ws / "drafts.jsonl"), "--corpus", str(ws / "corpus")]


def _verified(item_id: str) -> GoldenItem:
    return make_item(id=item_id, review=Review(status="accepted"), human_verified=True)


def test_build_refuses_until_quotas_met_then_writes_verified_only(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, card = workspace / "v1.jsonl", workspace / "README.md"
    build = ["build", "--out", str(out), "--card", str(card)]
    write_jsonl(workspace / "drafts.jsonl", [_verified("a"), make_item(id="b")])

    assert cli.main(_args(workspace, *build)) == 1
    assert "missing" in capsys.readouterr().out and not out.exists()

    assert cli.main(_args(workspace, *build, "--partial")) == 0
    assert [i.id for i in read_jsonl(out)] == ["a"]  # pending item "b" excluded
    assert "| factual | 1 | 100% |" in card.read_text(encoding="utf-8")


def test_build_rejects_evidence_that_does_not_match_corpus(workspace: Path) -> None:
    bad = _verified("a").model_copy(
        update={"evidence": (Evidence(doc_id="d.md", start=0, end=4, quote="Nope"),)}
    )
    write_jsonl(workspace / "drafts.jsonl", [bad, _verified("b")])
    args = _args(
        workspace,
        "build",
        "--out",
        str(workspace / "v1.jsonl"),
        "--card",
        str(workspace / "README.md"),
    )
    assert cli.main(args) == 1
    assert not (workspace / "v1.jsonl").exists()


def test_validate_and_stats(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = workspace / "v1.jsonl"
    write_jsonl(dataset, [_verified("a")])
    write_jsonl(workspace / "drafts.jsonl", [_verified("a"), make_item(id="b")])
    assert cli.main(_args(workspace, "validate", str(dataset))) == 0

    write_jsonl(dataset, [make_item(id="x")])  # unverified
    with pytest.raises(ValueError, match="unverified"):
        cli.main(_args(workspace, "validate", str(dataset)))
    assert cli.main(_args(workspace, "validate", str(dataset), "--allow-unverified")) == 0

    assert cli.main(_args(workspace, "stats")) == 0
    out = capsys.readouterr().out
    assert "factual" in out and "pending" in out


def test_commands_require_prepared_corpus(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()  # exists, but has no CORPUS_FINGERPRINT
    write_jsonl(tmp_path / "drafts.jsonl", [make_item()])
    with pytest.raises(SystemExit, match="not prepared"):
        cli.main(_args(tmp_path, "build"))
