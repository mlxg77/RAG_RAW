"""法律条文结构识别与法条切分。

把标准化后的文档（normalize_document() 的输出）切分成独立法条块，
并保留编 / 章 / 节和页码范围信息。
"""

import hashlib  # 计算 SHA-256，生成 chunk_id 和 text_hash
import re  # 正则匹配“第X条”“第X章”等结构
from collections import Counter  # 统计重复条号的出现次数
from dataclasses import dataclass, field  # 内部临时数据结构

from rag_law.schemas import (
    ArticleChunk,  # 输出：最终法条块
    LoadedDocument,  # 输入：加载并标准化后的文档
    SourceFormat,  # 文档格式类型：pdf / docx / markdown
    SplitResult,  # 输出：切分结果 + 质量统计
)


# 中文数字字符集：覆盖“〇零一二三四五六七八九十百千万两”及阿拉伯数字 0-9。
CHINESE_NUMBER = r"[〇零一二三四五六七八九十百千万两0-9]+"

# 条号必须位于行首，并且条号之后只能是空白或行尾。
# 这样不会把“第一条规定……”这种正文引用误判为新法条。
# “之X”后缀兼容“第十条之一”这类修正案条号。
ARTICLE_PATTERN = re.compile(
    rf"^(?P<article_no>"
    rf"第{CHINESE_NUMBER}条"
    rf"(?:之{CHINESE_NUMBER})?"
    rf")"
    rf"(?:\s+(?P<body>.*)|\s*)$"
)

# “第X编 名称”，如“第一编 总则”；名称可省略。
PART_PATTERN = re.compile(
    rf"^(?P<label>第{CHINESE_NUMBER}编)"
    rf"(?:\s+(?P<title>.+))?$"
)

# “第X章 名称”，如“第一章 总则”。
CHAPTER_PATTERN = re.compile(
    rf"^(?P<label>第{CHINESE_NUMBER}章)"
    rf"(?:\s+(?P<title>.+))?$"
)

# “第X节 名称”，如“第一节 一般规定”。
SECTION_PATTERN = re.compile(
    rf"^(?P<label>第{CHINESE_NUMBER}节)"
    rf"(?:\s+(?P<title>.+))?$"
)

# 单独的“附则”行，兼容“附 则”带空格的写法。
APPENDIX_PATTERN = re.compile(r"^附\s*则$")

# “（一）”“(2)”等列举项，兼容全角/半角括号。
LIST_ITEM_PATTERN = re.compile(
    rf"^[（(]{CHINESE_NUMBER}[）)]"
)

# 句末标点：用于判断上一段是否写完（PDF 跨页拼接的依据）。
PARAGRAPH_ENDINGS = ("。", "！", "？", "；", "：")

# 数据类：少写很多“样板代码”（__init__、__repr__、__eq__）
# frozen=True：使数据类不可修改
@dataclass(frozen=True)
class _SourceLine:
    """带原始位置的逻辑文本行。"""

    text: str  # 行文本内容
    page: int | None  # 来自第几页（PDF 有值，DOCX/MD 为 None）
    block_index: int  # 来自第几个文本块
    line_index: int  # 该文本块内部的第几行


@dataclass
class _BodySegment:
    """法条中的一个自然段或逻辑片段。"""

    text: str
    start_page: int | None
    end_page: int | None


@dataclass
class _ArticleDraft:
    """尚未转成最终 ArticleChunk 的法条。"""

    article_no: str  # 条号，如“第一条”
    part: str | None  # 所属编，如“第一编 总则”
    chapter: str | None  # 所属章
    section: str | None  # 所属节
    start_page: int | None  # 起始页
    end_page: int | None  # 结束页
    # default_factory 保证每个实例都有独立的列表
    segments: list[_BodySegment] = field(default_factory=list)


def _strip_markdown_heading(line: str) -> str:
    """去掉 Markdown 标题前缀。

    例如：

        ## 第一节 一般规定

    转换成：

        第一节 一般规定
    """

    # ^#{1,6}\s* 匹配行首 1-6 个 # 号及其后的空白字符
    return re.sub(r"^#{1,6}\s*", "", line).strip()


def _iter_source_lines(
    document: LoadedDocument,
) -> list[_SourceLine]:
    """把文档块展开成携带页码的逻辑文本行。"""

    source_lines: list[_SourceLine] = []

    for block in document.blocks:
        # 按换行符拆行，line_index 是块内行号
        for line_index, raw_line in enumerate(
            block.text.splitlines()
        ):
            line = raw_line.strip()

            # Markdown 标题行去掉 # 前缀，统一成纯文本
            if document.source_format == "markdown":
                line = _strip_markdown_heading(line)

            # 空行不产生文本行
            if not line:
                continue

            source_lines.append(
                _SourceLine(
                    text=line,
                    page=block.page,
                    block_index=block.block_index,
                    line_index=line_index,
                )
            )

    return source_lines


def _heading_value(match: re.Match[str]) -> str:
    """把标题编号和标题名称拼成标准形式。"""

    label = match.group("label")
    title = match.group("title")

    # 只有编号没有名称时，直接返回编号
    if not title:
        return label

    # 连续空格压缩为一个，但不删除标题中的正常空格。
    title = re.sub(r"\s+", " ", title).strip()

    return f"{label} {title}"


def _find_body_start(
    source_lines: list[_SourceLine],
) -> int | None:
    """寻找法律正文第一条的位置。

    法律通常从“第一条”开始。如果目录里也列出了“第一条”，
    正文中的“第一条”会出现在后面，因此选择最后一个“第一条”。

    如果文档没有“第一条”，则退化为第一个可识别的条号。
    """

    all_article_indexes: list[int] = []
    first_article_indexes: list[int] = []

    for index, source_line in enumerate(source_lines):
        match = ARTICLE_PATTERN.fullmatch(source_line.text)

        if match is None:
            continue

        # 记录所有条号的位置
        all_article_indexes.append(index)

        # 单独记录“第一条”的位置（目录区也可能出现）
        if match.group("article_no") == "第一条":
            first_article_indexes.append(index)

    # 取最后一个“第一条”：前面的通常属于目录，正文中的在后面
    if first_article_indexes:
        return first_article_indexes[-1]

    # 没有“第一条”时，退化为第一个可识别的条号
    if all_article_indexes:
        return all_article_indexes[0]

    return None


def _append_body_text(
    draft: _ArticleDraft,
    *,
    text: str,
    page: int | None,
    source_format: SourceFormat,
) -> None:
    """向当前法条追加一段正文。

    对 PDF，如果法条跨页，并且上一页末尾没有结束标点，
    则把下一页开头接到上一段末尾。
    """

    text = text.strip()

    if not text:
        return

    should_join_across_page = (
        source_format == "pdf"
        and page is not None
        and draft.end_page is not None
        and page == draft.end_page + 1
        and bool(draft.segments)
        and not draft.segments[-1].text.endswith(
            PARAGRAPH_ENDINGS
        )
        and LIST_ITEM_PATTERN.match(text) is None
    )

    if should_join_across_page:
        last_segment = draft.segments[-1]
        last_segment.text += text
        last_segment.end_page = page
    else:
        draft.segments.append(
            _BodySegment(
                text=text,
                start_page=page,
                end_page=page,
            )
        )

    if page is not None:
        if draft.start_page is None:
            draft.start_page = page

        if draft.end_page is None:
            draft.end_page = page
        else:
            draft.end_page = max(draft.end_page, page)

def _draft_is_empty(draft: _ArticleDraft) -> bool:
    """判断法条是否没有正文。"""

    return (
        not draft.segments
        or not any(
            segment.text.strip()
            for segment in draft.segments
        )
    )

def _sha256(value: str) -> str:
    """生成带算法前缀的 SHA-256。"""

    # 先按 UTF-8 编码成字节串，再取十六进制摘要
    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    # 前缀记录算法名，便于将来更换算法时区分
    return f"sha256:{digest}"
    """把内部法条对象转换成最终数据结构。"""

    # 多个自然段用换行拼接成完整条文
    text = "\n".join(
        segment.text
        for segment in draft.segments
    ).strip()

    # 内容指纹：用于判断条文内容是否变化
    text_hash = _sha256(text)

    # occurrence 用于处理原文中意外出现的重复条号。
    # 正常情况下每个条号 occurrence 都是 1。
    # 用 \0 拼接，避免字段边界产生歧义。
    identity = "\0".join(
        [
            document.law_id,
            draft.article_no,
            str(occurrence),
        ]
    )

    # 条文身份标识：由 法律ID + 条号 + 出现序号 共同决定
    chunk_id = _sha256(identity)

    return ArticleChunk(
        chunk_id=chunk_id,
        law_id=document.law_id,
        law_name=document.law_name,
        part=draft.part,
        chapter=draft.chapter,
        section=draft.section,
        article_no=draft.article_no,
        paragraph_no=None,  # 暂不按款拆分
        text=text,
        source_file=document.source_file,
        page=draft.start_page,
        end_page=draft.end_page,
        text_hash=text_hash,
    )

# 拆分原则：
# 一条法条不超过 1200 字符
#     → 一个 chunk，paragraph_no = null

# 一条法条超过 1200 字符，且有多个自然段
#     → 按自然段组合成多个 chunk
#     → paragraph_no = 1、2、3……

# 单个自然段自己就超过 1200 字符
#     → 保留整个自然段
#     → 不从一句话中间硬切
#     → 在质量报告中标记 oversized

def _split_segments(
    segments: list[_BodySegment],
    *,
    max_chars: int,
) -> list[list[_BodySegment]]:
    """按自然段组合超长法条。

    不从自然段中间截断。单个自然段本身超过 max_chars 时，
    保留完整自然段，因此最终 chunk 可能略大于阈值。
    """

    full_text = "\n".join(
        segment.text
        for segment in segments
    )

    if len(full_text) <= max_chars:
        return [segments]

    groups: list[list[_BodySegment]] = []
    current_group: list[_BodySegment] = []
    current_length = 0

    for segment in segments:
        separator_length = 1 if current_group else 0
        projected_length = (
            current_length
            + separator_length
            + len(segment.text)
        )

        if current_group and projected_length > max_chars:
            groups.append(current_group)
            current_group = []
            current_length = 0

        if current_group:
            current_length += 1

        current_group.append(segment)
        current_length += len(segment.text)

    if current_group:
        groups.append(current_group)

    return groups


def _group_page_range(
    segments: list[_BodySegment],
) -> tuple[int | None, int | None]:
    """取得一组自然段的开始页和结束页。"""

    start_page = next(
        (
            segment.start_page
            for segment in segments
            if segment.start_page is not None
        ),
        None,
    )

    end_page = next(
        (
            segment.end_page
            for segment in reversed(segments)
            if segment.end_page is not None
        ),
        None,
    )

    return start_page, end_page


def _draft_to_chunks(
    draft: _ArticleDraft,
    *,
    document: LoadedDocument,
    occurrence: int,
    max_chars: int,
) -> list[ArticleChunk]:
    """把一个法条转换成一个或多个检索块。"""

    segment_groups = _split_segments(
        draft.segments,
        max_chars=max_chars,
    )

    was_split = len(segment_groups) > 1
    chunks: list[ArticleChunk] = []

    for group_index, segment_group in enumerate(
        segment_groups,
        start=1,
    ):
        text = "\n".join(
            segment.text
            for segment in segment_group
        ).strip()

        text_hash = _sha256(text)

        identity_parts = [
            document.law_id,
            draft.article_no,
            str(occurrence),
        ]

        # 未拆分法条维持原来的 ID 算法。
        # 只有真正拆分后才加入子块序号。
        if was_split:
            identity_parts.append(str(group_index))

        chunk_id = _sha256(
            "\0".join(identity_parts)
        )

        if was_split:
            page, end_page = _group_page_range(
                segment_group
            )
            paragraph_no: int | None = group_index
        else:
            page = draft.start_page
            end_page = draft.end_page
            paragraph_no = None

        chunks.append(
            ArticleChunk(
                chunk_id=chunk_id,
                law_id=document.law_id,
                law_name=document.law_name,
                part=draft.part,
                chapter=draft.chapter,
                section=draft.section,
                article_no=draft.article_no,
                paragraph_no=paragraph_no,
                text=text,
                source_file=document.source_file,
                page=page,
                end_page=end_page,
                text_hash=text_hash,
            )
        )

    return chunks


def split_articles(
    document: LoadedDocument,
    *,
    max_chars: int = 1200,
) -> SplitResult:
    """把标准化后的整部法律切分成独立法条。

    参数必须是 normalize_document() 的输出。
    核心是一个逐行扫描的状态机：遇到新条号就切出一条法条。
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")

    # 展开成带位置信息的行，并定位正文起点（跳过目录区）
    source_lines = _iter_source_lines(document)
    body_start = _find_body_start(source_lines)

    # 一个条号都没找到，直接返回空结果 + 警告
    if body_start is None:
        return SplitResult(
            chunks=[],
            article_count=0,
            warnings=[
                f"未识别到任何法条：{document.source_file}"
            ],
        )

    # 记录当前所处的编 / 章 / 节，随扫描不断更新
    current_part: str | None = None
    current_chapter: str | None = None
    current_section: str | None = None

    # 当前正在加工的法条草稿 + 已完成的草稿列表
    current_article: _ArticleDraft | None = None
    drafts: list[_ArticleDraft] = []

    # 把正在加工中的当前法条，存进已完成的法条列表，然后清空当前状态，准备接收下一条。
    def finish_current_article() -> None:
        nonlocal current_article

        if current_article is not None:
            drafts.append(current_article)
            current_article = None

    # 主循环：逐行判断每一行属于哪种结构
    for index, source_line in enumerate(source_lines):
        line = source_line.text

        part_match = PART_PATTERN.fullmatch(line)

        # 1. 遇到“第X编”标题
        if part_match is not None:
            if index >= body_start:
                finish_current_article()

            current_part = _heading_value(part_match)

            # 进入新编时，旧章和旧节不再有效。
            current_chapter = None
            current_section = None
            continue

        chapter_match = CHAPTER_PATTERN.fullmatch(line)

        # 2. 遇到“第X章”标题
        if chapter_match is not None:
            if index >= body_start:
                finish_current_article()

            current_chapter = _heading_value(chapter_match)

            # 进入新章时，旧节不再有效。
            current_section = None
            continue

        section_match = SECTION_PATTERN.fullmatch(line)

        # 3. 遇到“第X节”标题
        if section_match is not None:
            if index >= body_start:
                finish_current_article()

            current_section = _heading_value(section_match)
            continue

        # 4. 遇到“附则”
        if APPENDIX_PATTERN.fullmatch(line):
            if index >= body_start:
                finish_current_article()

            # “附则”通常位于编章体系之外。
            current_part = None
            current_chapter = "附则"
            current_section = None
            continue

        article_match = ARTICLE_PATTERN.fullmatch(line)

        # 正文开始前出现的条号属于目录或其他前置内容。
        if index < body_start:
            continue

        # 5. 遇到新条号：核心切割点，结束上一条、开启新条草稿
        if article_match is not None:
            finish_current_article()

            current_article = _ArticleDraft(
                article_no=article_match.group("article_no"),
                part=current_part,
                chapter=current_chapter,
                section=current_section,
                start_page=source_line.page,
                end_page=source_line.page,
            )

            # 条号后同一行的正文也要收进来
            body = article_match.group("body") or ""

            _append_body_text(
                current_article,
                text=body,
                page=source_line.page,
                source_format=document.source_format,
            )
            continue

        # 正文开始后，普通文本归入当前法条。
        if current_article is not None:
            _append_body_text(
                current_article,
                text=line,
                page=source_line.page,
                source_format=document.source_format,
            )

    # 循环结束，收尾最后一条法条
    finish_current_article()

    # 统计每个条号的出现次数，用于发现重复条号
    article_number_counts = Counter(
        draft.article_no
        for draft in drafts
    )

    # 找出出现次数大于 1 的条号（通常意味着原文本身有重复）
    duplicate_articles = sorted(
        article_no
        for article_no, count
        in article_number_counts.items()
        if count > 1
    )

    # 找出空条：没有正文段落，或所有段落都是空白
    empty_articles = [
        draft.article_no
        for draft in drafts
        if _draft_is_empty(draft)
    ]

    # 逐条转换：空条跳过，重复条号用 occurrence 区分
    occurrence_counts: Counter[str] = Counter()
    chunks: list[ArticleChunk] = []

    for draft in drafts:
        occurrence_counts[draft.article_no] += 1

        if _draft_is_empty(draft):
            continue

        chunks.extend(
            _draft_to_chunks(
                draft,
                document=document,
                occurrence=occurrence_counts[draft.article_no],
                max_chars=max_chars,
            )
        )

    warnings: list[str] = []

    if empty_articles:
        warnings.append(
            f"发现 {len(empty_articles)} 个空条"
        )

    if duplicate_articles:
        warnings.append(
            f"发现 {len(duplicate_articles)} 个重复条号"
        )

    oversized_chunks = [
        chunk
        for chunk in chunks
        if len(chunk.text) > max_chars
    ]

    if oversized_chunks:
        warnings.append(
            f"发现 {len(oversized_chunks)} 个无法按自然段继续拆分的超长块"
        )

    return SplitResult(
        chunks=chunks,
        article_count=len(drafts),
        empty_articles=empty_articles,
        duplicate_articles=duplicate_articles,
        warnings=warnings,
    )
