from pathlib import Path

import pytest

from rag_law.catalog import load_law_options


def test_load_law_options_preserves_manifest_order(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "- law_id: first\n  law_name: 第一部法\n"
        "- law_id: second\n  law_name: 第二部法\n",
        encoding="utf-8",
    )

    options = load_law_options(manifest)

    assert [option.law_id for option in options] == ["first", "second"]
    assert options[1].law_name == "第二部法"


def test_load_law_options_rejects_missing_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("- law_id: incomplete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="law_id 或 law_name"):
        load_law_options(manifest)
