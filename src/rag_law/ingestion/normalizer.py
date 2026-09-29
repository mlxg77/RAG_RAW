"""法律文本标准化。

职责：
- 删除不可见字符
- 统一空白符和换行符
- 删除 PDF 中重复出现的页眉、页脚和页码
- 修复 PDF 因视觉排版造成的句中换行

本模块不负责识别目录、章节或法条。
"""

import math
import re
from collections import Counter

from rag_law.schemas import LoadedDocument, SourceBlock


# 中文法律条号可能包含中文数字、阿拉伯数字以及“之一”等形式。
CHINESE_NUMBER = r"[〇零一二三四五六七八九十百千万两0-9]+"

# 匹配“第一编”“第二章”“第三节”“第一百零一条”
# Markdown 标题开头可能带 #、## 等标记。
STRUCTURE_PATTERN = re.compile(
    rf"^(?:#{{1,6}}\s*)?"
    rf"第{CHINESE_NUMBER}(?:编|章|节|条)"
    rf"(?:之{CHINESE_NUMBER})?"
    rf"(?:\s|$)"
)

# 匹配“（一）”“（二）”“(1)”等列举项。
LIST_ITEM_PATTERN = re.compile(
    rf"^[（(](?:{CHINESE_NUMBER})[）)]"
)

# 单独占一行的 PDF 页码，例如：
# 12
# - 12 -
# — 12 —
# 第12页
PAGE_NUMBER_PATTERN = re.compile(
    r"^(?:第\s*)?"
    r"[-—–]?\s*\d+\s*[-—–]?"
    r"(?:\s*页)?$"
)

# 这些字符通常不可见，但会污染正则匹配、哈希和检索文本。
INVISIBLE_CHARACTER_TRANSLATION = str.maketrans(
    {
        "\u200b": None,  # 零宽空格
        "\u200c": None,  # 零宽非连接符
        "\u200d": None,  # 零宽连接符
        "\u2060": None,  # 单词连接符
        "\ufeff": None,  # BOM / 零宽不换行空格
        "\u00ad": None,  # 软连字符
    }
)

# 匹配所有水平空白符，包括空格、制表符、换页符和垂直制表符。
HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[ \t\f\v]+")

# 如果上一行以这些标点结尾，下一行通常是新的自然段或列举项，
# 而不是上一行的视觉换行。
PARAGRAPH_ENDINGS = ("。", "！", "？", "；", "：")


def normalize_text(text: str) -> str:
    """执行与文件格式无关的基础字符清洗。

    注意：不使用 unicodedata.normalize("NFKC", text)，因为 NFKC
    会修改部分字符形态。法律原文只做必要清洗，不做过度转换。
    """
    # translate() 是 Python 字符串 str 的方法，用来按照“字符映射表”批量替换或删除字符。
    text = text.translate(INVISIBLE_CHARACTER_TRANSLATION)

    # 统一不同平台的换行符。
    text = text.replace("\r\n", "\n") # 替换回车换行符为换行符
    text = text.replace("\r", "\n") # 替换回车符为换行符

    # 全角空格和不间断空格统一为普通空格。
    text = text.replace("\u3000", " ") # 全角空格统一为空格
    text = text.replace("\u00a0", " ") # 不间断空格统一为空格

    cleaned_lines: list[str] = []
    # 用来记录“上一行是不是空行”。
    previous_line_was_blank = False

    for raw_line in text.split("\n"):
        # 清理当前行里的空白字符，并去掉首尾空格。
        line = HORIZONTAL_WHITESPACE_PATTERN.sub(" ", raw_line).strip()

        if not line:
            # 连续多个空行最多保留一个。
            if cleaned_lines and not previous_line_was_blank:
                cleaned_lines.append("")

            previous_line_was_blank = True
            continue

        cleaned_lines.append(line)
        previous_line_was_blank = False

    # 删除末尾残留的空行。
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


def _margin_key(line: str) -> str:
    """生成页眉页脚比较键。

    页眉中的普通空格不应影响重复判断，例如：
    “中华人民共和国劳动合同法”
    和
    “中华人民共和国 劳动合同法”
    应视为同一个候选页眉。
    """

    return re.sub(r"\s+", "", line)


def _non_empty_line_indexes(lines: list[str]) -> list[int]:
    """返回所有非空行的位置。"""

    return [
        index
        for index, line in enumerate(lines)
        if line.strip()
    ]


def _margin_indexes(lines: list[str]) -> set[int]:
    """返回页面顶部两行和底部两行的索引。

    只有页面边缘位置才允许被判定成页眉页脚，避免误删正文中
    偶然重复的短句。
    """

    non_empty_indexes = _non_empty_line_indexes(lines)

    return set(
        non_empty_indexes[:2]
        + non_empty_indexes[-2:]
    )


def _find_repeated_margin_keys(
    pages: list[list[str]],
) -> set[str]:
    """寻找在多个 PDF 页面边缘重复出现的文本。

    对三页以上的文档，至少重复三页才删除；同时还要求重复页数
    达到总页数的 30%，避免把偶然出现在两页边缘的正文删掉。
    """

    if not pages:
        return set()

    if len(pages) <= 2:
        threshold = 2
    else:
        threshold = max(3, math.ceil(len(pages) * 0.3))

    page_occurrences: Counter[str] = Counter()

    for lines in pages:
        keys_on_this_page: set[str] = set()

        for index in _margin_indexes(lines):
            key = _margin_key(lines[index])

            # 页眉页脚通常较短。限制长度可以避免把正文长句当成页眉。
            if 2 <= len(key) <= 80:
                keys_on_this_page.add(key)

        # 同一页重复两次仍然只计一次。
        page_occurrences.update(keys_on_this_page)

    return {
        key
        for key, count in page_occurrences.items()
        if count >= threshold
    }


def _remove_pdf_margins(
    lines: list[str],
    repeated_margin_keys: set[str],
) -> list[str]:
    """删除一页中的重复页眉、页脚和单独页码。"""

    margin_indexes = _margin_indexes(lines)
    kept_lines: list[str] = []

    for index, line in enumerate(lines):
        if index not in margin_indexes:
            kept_lines.append(line)
            continue

        key = _margin_key(line)

        if key in repeated_margin_keys:
            continue

        if PAGE_NUMBER_PATTERN.fullmatch(line):
            continue

        kept_lines.append(line)

    return kept_lines


def _starts_new_logical_line(
    previous_line: str,
    current_line: str,
) -> bool:
    """判断当前 PDF 行是否应开始一个新的逻辑段落。"""

    # 编、章、节、条必须独立开始，不能粘到上一行末尾。
    if STRUCTURE_PATTERN.match(current_line):
        return True

    # “（一）”等列举项应该独立成段。
    if LIST_ITEM_PATTERN.match(current_line):
        return True

    # 上一行以完整句或提示性标点结束时，保留换行。
    if previous_line.endswith(PARAGRAPH_ENDINGS):
        return True

    return False


def _repair_pdf_lines(lines: list[str]) -> str:
    """把 PDF 的视觉换行修复成逻辑段落。

    示例：

        第一条 劳动合同期限三个月以上不满一年
        的，试用期不得超过一个月。

    修复为：

        第一条 劳动合同期限三个月以上不满一年的，试用期不得超过一个月。
    """

    logical_lines: list[str] = []
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer

        if buffer:
            logical_lines.append(buffer)
            buffer = ""

    for line in lines:
        if not line:
            flush_buffer()
            continue

        if not buffer:
            buffer = line
            continue

        if _starts_new_logical_line(buffer, line):
            flush_buffer()
            buffer = line
        else:
            # 句中换行直接拼接，不额外添加空格。
            buffer += line

    flush_buffer()

    return "\n".join(logical_lines)


def _normalize_pdf_blocks(
    blocks: list[SourceBlock],
) -> list[SourceBlock]:
    """标准化 PDF 页面，并清除重复页边文本。"""

    normalized_page_lines: list[list[str]] = []

    for block in blocks:
        normalized_text = normalize_text(block.text)
        normalized_page_lines.append(normalized_text.split("\n"))

    repeated_margin_keys = _find_repeated_margin_keys(
        normalized_page_lines
    )

    normalized_blocks: list[SourceBlock] = []

    for block, lines in zip(
        blocks,
        normalized_page_lines,
        strict=True,
    ):
        lines_without_margins = _remove_pdf_margins(
            lines,
            repeated_margin_keys,
        )

        repaired_text = _repair_pdf_lines(lines_without_margins)

        if not repaired_text:
            continue

        normalized_blocks.append(
            block.model_copy(
                update={"text": repaired_text}
            )
        )

    return normalized_blocks


def _normalize_non_pdf_blocks(
    blocks: list[SourceBlock],
) -> list[SourceBlock]:
    """标准化 DOCX 或 Markdown 文本块。

    DOCX 段落边界和 Markdown 行边界本身具有意义，因此这里不做
    PDF 式的跨行合并。
    """

    normalized_blocks: list[SourceBlock] = []

    for block in blocks:
        normalized_text = normalize_text(block.text)

        if not normalized_text:
            continue

        normalized_blocks.append(
            block.model_copy(
                update={"text": normalized_text}
            )
        )

    return normalized_blocks


def normalize_document(
    document: LoadedDocument,
) -> LoadedDocument:
    """标准化整部法律，同时保留来源和位置元数据。"""

    if document.source_format == "pdf":
        normalized_blocks = _normalize_pdf_blocks(
            document.blocks
        )
    else:
        normalized_blocks = _normalize_non_pdf_blocks(
            document.blocks
        )

    if not normalized_blocks:
        raise ValueError(
            f"文档标准化后没有剩余文本：{document.source_file}"
        )

    return document.model_copy(
        update={"blocks": normalized_blocks}
    )