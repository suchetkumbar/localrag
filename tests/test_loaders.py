from pathlib import Path

import pytest

from rag.loaders import load_corpus, normalise_text


def test_normalise_crlf_and_cr() -> None:
    assert normalise_text("a\r\nb\rc\n") == "a\nb\nc\n"


def test_load_corpus_ids_titles_and_order(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "two.md").write_bytes(b"# Second Doc\r\n\r\nBody.\r\n")
    (tmp_path / "a.md").write_text("No heading here.\n", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("not markdown", encoding="utf-8")

    docs = load_corpus(tmp_path, ["**/*.md"])

    assert [d.doc_id for d in docs] == ["a.md", "b/two.md"]  # POSIX ids, sorted, empty dropped
    assert docs[0].title == "a"  # falls back to file stem
    assert docs[1].title == "Second Doc"
    assert "\r" not in docs[1].text


def test_missing_corpus_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")
