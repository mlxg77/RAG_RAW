"""三种法律文件 Loader 的自动化测试。"""

from pathlib import Path

import pytest

from rag_law.ingestion.loaders import load_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def joined_text(document) -> str:
    return "\n".join(block.text for block in document.blocks)


def test_load_pdf() -> None:
    relative_path = "data/raw/中华人民共和国劳动合同法.pdf"

    document = load_document(
        PROJECT_ROOT / relative_path,
        law_id="cn_labor_contract_law",
        law_name="中华人民共和国劳动合同法",
        source_file=relative_path,
    )

    assert document.source_format == "pdf"
    assert document.source_file == relative_path
    assert document.page_count is not None
    assert document.page_count >= 10
    assert len(document.blocks) == document.page_count
    assert all(block.page is not None for block in document.blocks)

    text = joined_text(document)
    assert "中华人民共和国劳动合同法" in text
    assert "第一条" in text
    assert "第十条" in text


def test_load_docx() -> None:
    relative_path = "data/raw/中华人民共和国刑法.docx"

    document = load_document(
        PROJECT_ROOT / relative_path,
        law_id="cn_criminal_law",
        law_name="中华人民共和国刑法",
        source_file=relative_path,
    )

    assert document.source_format == "docx"
    assert document.source_file == relative_path
    assert document.page_count is None
    assert len(document.blocks) > 100
    assert all(block.page is None for block in document.blocks)

    text = joined_text(document)
    assert "中华人民共和国刑法" in text
    assert "第一编" in text
    assert "第一条" in text


def test_load_markdown() -> None:
    relative_path = "data/raw/中华人民共和国道路交通安全法.md"

    document = load_document(
        PROJECT_ROOT / relative_path,
        law_id="cn_road_traffic_safety_law",
        law_name="中华人民共和国道路交通安全法",
        source_file=relative_path,
    )

    assert document.source_format == "markdown"
    assert document.source_file == relative_path
    assert document.page_count is None
    assert len(document.blocks) > 100
    assert all(block.page is None for block in document.blocks)

    text = joined_text(document)
    assert "中华人民共和国道路交通安全法" in text
    assert "第一章" in text
    assert "第一条" in text


def test_missing_file_raises_error() -> None:
    with pytest.raises(FileNotFoundError, match="不存在"):
        load_document(
            PROJECT_ROOT / "data" / "raw" / "不存在.pdf",
            law_id="missing",
            law_name="不存在",
        )


def test_unsupported_format_raises_error(tmp_path: Path) -> None:
    source = tmp_path / "law.txt"
    source.write_text("第一条 测试。", encoding="utf-8")

    with pytest.raises(ValueError, match="不支持的文件格式"):
        load_document(
            source,
            law_id="test_law",
            law_name="测试法",
        )