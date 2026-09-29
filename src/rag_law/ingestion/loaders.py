"""PDF、DOCX 和 Markdown 原始文本加载器。

本模块只负责提取文本和原始位置，不负责：
- 文本清洗
- 目录过滤
- 法条识别
- 文本切分
"""

from pathlib import Path

import pymupdf
from docx import Document

from rag_law.schemas import LoadedDocument, SourceBlock


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md"}


def _source_name(path: Path, source_file: str | None) -> str:
    """优先使用 manifest 中保存的相对路径。"""

    return source_file if source_file is not None else path.as_posix()


def load_pdf(
    path: Path,
    *,
    law_id: str,
    law_name: str,
    source_file: str | None = None,
) -> LoadedDocument:
    """按页提取 PDF 文本。

    一页作为一个 SourceBlock，可以让后面的 splitter 追踪：
    - 法条开始页
    - 法条结束页
    - 跨页法条
    """

    blocks: list[SourceBlock] = []

    with pymupdf.open(path) as document:
        page_count = len(document)

        for page_index, page in enumerate(document):
            text = page.get_text("text", sort=True)

            blocks.append(
                SourceBlock(
                    text=text,
                    page=page_index + 1,
                    block_index=page_index,
                )
            )

    visible_character_count = sum(
        len("".join(block.text.split())) # 去除空格
        for block in blocks
    )

    if visible_character_count < 50:
        raise ValueError(
            f"PDF 几乎没有可提取文本，可能是扫描件，需要 OCR：{path}"
        )

    return LoadedDocument(
        law_id=law_id,
        law_name=law_name,
        source_file=_source_name(path, source_file),
        source_format="pdf",
        page_count=page_count,
        blocks=blocks,
    )


def load_docx(
    path: Path,
    *,
    law_id: str,
    law_name: str,
    source_file: str | None = None,
) -> LoadedDocument:
    """按 Word 段落提取 DOCX 文本。"""

    document = Document(path)
    blocks: list[SourceBlock] = []

    for paragraph_index, paragraph in enumerate(document.paragraphs):
        text = paragraph.text

        # 遇到空格和换行符则跳过
        if not text.strip():
            continue

        blocks.append(
            SourceBlock(
                text=text,
                page=None,
                block_index=paragraph_index,
            )
        )

    if not blocks:
        raise ValueError(f"DOCX 中没有可提取的非空段落：{path}")

    return LoadedDocument(
        law_id=law_id,
        law_name=law_name,
        source_file=_source_name(path, source_file),
        source_format="docx",
        page_count=None,
        blocks=blocks,
    )


def load_markdown(
    path: Path,
    *,
    law_id: str,
    law_name: str,
    source_file: str | None = None,
) -> LoadedDocument:
    """按非空行提取 Markdown 文本。"""

    content = path.read_text(encoding="utf-8")
    blocks: list[SourceBlock] = []

    for line_index, line in enumerate(content.splitlines()):
        if not line.strip():
            continue

        blocks.append(
            SourceBlock(
                text=line,
                page=None,
                block_index=line_index,
            )
        )

    if not blocks:
        raise ValueError(f"Markdown 文件为空：{path}")

    return LoadedDocument(
        law_id=law_id,
        law_name=law_name,
        source_file=_source_name(path, source_file),
        source_format="markdown",
        page_count=None,
        blocks=blocks,
    )


def load_document(
    path: Path,
    *,
    law_id: str,
    law_name: str,
    source_file: str | None = None,
) -> LoadedDocument:
    """根据文件扩展名分发到对应 Loader。"""

    path = path.resolve() # resolve()把一个路径解析成“绝对路径”

    if not path.is_file(): # is_file()判断路径是否为文件
        raise FileNotFoundError(f"原始法律文件不存在：{path}")

    suffix = path.suffix.lower() # suffix()获取文件后缀名，lower()转换为小写

    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"不支持的文件格式 {suffix!r}：{path}；"
            f"当前支持 {sorted(SUPPORTED_SUFFIXES)}"
        )

    if suffix == ".pdf":
        return load_pdf(
            path,
            law_id=law_id,
            law_name=law_name,
            source_file=source_file,
        )

    if suffix == ".docx":
        return load_docx(
            path,
            law_id=law_id,
            law_name=law_name,
            source_file=source_file,
        )

    return load_markdown(
        path,
        law_id=law_id,
        law_name=law_name,
        source_file=source_file,
    )