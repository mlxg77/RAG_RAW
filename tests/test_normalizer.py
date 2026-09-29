"""文本标准化测试。"""

from pathlib import Path

from rag_law.ingestion.loaders import load_document
from rag_law.ingestion.normalizer import (
    normalize_document,
    normalize_text,
)
from rag_law.schemas import LoadedDocument, SourceBlock


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_pdf_document(
    page_texts: list[str],
) -> LoadedDocument:
    """构造一个不依赖真实文件的 PDF 文档对象。"""

    return LoadedDocument(
        law_id="test_law",
        law_name="测试法",
        source_file="data/raw/测试法.pdf",
        source_format="pdf",
        page_count=len(page_texts),
        blocks=[
            SourceBlock(
                text=text,
                page=index + 1,
                block_index=index,
            )
            for index, text in enumerate(page_texts)
        ],
    )


def test_normalize_basic_characters() -> None:
    source = (
        "\ufeff  第一\u3000条\t 测\u200b试。\r\n"
        "\r\n"
        "  第二行\u00a0文字\u2060  "
    )

    result = normalize_text(source)

    assert result == "第一 条 测试。\n\n第二行 文字"


def test_repair_pdf_visual_line_break() -> None:
    document = make_pdf_document(
        [
            (
                "第一条　劳动合同期限三个月以上不满一年\n"
                "的，试用期不得超过一个月。\n"
                "第二条　这是下一条。"
            )
        ]
    )

    normalized = normalize_document(document)

    assert normalized.blocks[0].text == (
        "第一条 劳动合同期限三个月以上不满一年的，"
        "试用期不得超过一个月。\n"
        "第二条 这是下一条。"
    )


def test_preserve_pdf_list_structure() -> None:
    document = make_pdf_document(
        [
            (
                "第一条　应当符合下列条件：\n"
                "（一）第一项；\n"
                "（二）第二项。"
            )
        ]
    )

    normalized = normalize_document(document)

    assert normalized.blocks[0].text == (
        "第一条 应当符合下列条件：\n"
        "（一）第一项；\n"
        "（二）第二项。"
    )


def test_remove_repeated_pdf_header_footer_and_page_number() -> None:
    document = make_pdf_document(
        [
            "测试法\n第一章　总则\n第一条　第一页内容。\n— 1 —",
            "测试法\n第二条　第二页内容。\n— 2 —",
            "测试法\n第三条　第三页内容。\n— 3 —",
            "测试法\n第四条　第四页内容。\n— 4 —",
        ]
    )

    normalized = normalize_document(document)
    text = "\n".join(
        block.text
        for block in normalized.blocks
    )

    assert "测试法" not in text
    assert "— 1 —" not in text
    assert "— 2 —" not in text
    assert "第一章 总则" in text
    assert "第一条 第一页内容。" in text
    assert "第四条 第四页内容。" in text


def test_non_pdf_blocks_are_not_merged() -> None:
    document = LoadedDocument(
        law_id="test_law",
        law_name="测试法",
        source_file="data/raw/测试法.docx",
        source_format="docx",
        page_count=None,
        blocks=[
            SourceBlock(
                text="第一条\u3000第一款。",
                page=None,
                block_index=10,
            ),
            SourceBlock(
                text="第二款。\u200b",
                page=None,
                block_index=11,
            ),
        ],
    )

    normalized = normalize_document(document)

    assert len(normalized.blocks) == 2
    assert normalized.blocks[0].text == "第一条 第一款。"
    assert normalized.blocks[1].text == "第二款。"
    assert normalized.blocks[0].block_index == 10
    assert normalized.blocks[1].block_index == 11


def test_normalization_does_not_mutate_input_document() -> None:
    document = make_pdf_document(
        ["第一条\u3000测\u200b试。"]
    )

    normalized = normalize_document(document)

    assert document.blocks[0].text == "第一条\u3000测\u200b试。"
    assert normalized.blocks[0].text == "第一条 测试。"


def test_real_pdf_invisible_characters_are_removed() -> None:
    relative_path = "data/raw/中华人民共和国劳动合同法.pdf"

    loaded = load_document(
        PROJECT_ROOT / relative_path,
        law_id="cn_labor_contract_law",
        law_name="中华人民共和国劳动合同法",
        source_file=relative_path,
    )

    normalized = normalize_document(loaded)

    text = "\n".join(
        block.text
        for block in normalized.blocks
    )

    assert "\u200b" not in text
    assert "\ufeff" not in text
    assert "第一条" in text
    assert "第十条" in text

    # 标准化不能破坏 PDF 页码映射。
    assert all(
        block.page is not None
        for block in normalized.blocks
    )