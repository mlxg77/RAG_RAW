"""按条号精确补召回。

「第几条」是结构化查询，不是语义相似度问题：十五条法律里「第X条」
的表述几乎一模一样，Dense 的分数会被压成一团（实测 0.71 附近十几
条并列，名次基本随机），而 RRF 的固定权重又让 BM25 的命中排不进候
选池（BM25 第 1 名的贡献低于 Dense 第 10 名）。

所以这里在融合之后直接按 (法律, 条号) 把那条法条补进候选集：
- 不改任何全局权重或候选池大小，其它查询的行为完全不变；
- 补进来的只是候选，最终去留仍由精排决定。
"""

import logging
import re
from pathlib import Path

from rag_law.retrieval.common import (
    DEFAULT_ARTICLES_PATH,
    load_articles,
    make_retrieval_result,
)
from rag_law.schemas import ArticleChunk, RetrievalResult


logger = logging.getLogger(__name__)

NATIONAL_PREFIX = "中华人民共和国"
# 一次最多补两条：够覆盖「第二条」「第一百二十条之一」这类，
# 又不至于把候选池撑大、抬高精排成本。
DEFAULT_ANCHOR_LIMIT = 2

_CN_DIGITS = "零一二三四五六七八九"
_CN_UNITS = ("", "十", "百", "千")
_FULL_WIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
# 同时接受「第2条」「第 20 条」和「第一百零五条」，以及「之一」后缀。
_ARTICLE_REFERENCE_PATTERN = re.compile(
    r"第\s*(\d+|[零一二三四五六七八九十百千]+)\s*条"
    r"(?:之\s*([一二三四五六七八九]|\d))?"
)


def to_chinese_article_number(value: int) -> str:
    """把条号数字转成与索引一致的中文形态，101 -> 一百零一。"""

    if value <= 0 or value > 9999:
        return str(value)

    digits = [int(char) for char in str(value)]
    parts: list[str] = []
    pending_zero = False

    for index, digit in enumerate(digits):
        if digit == 0:
            pending_zero = True
            continue

        if pending_zero and parts:
            parts.append("零")
        pending_zero = False

        unit = _CN_UNITS[len(digits) - index - 1]
        parts.append(f"{_CN_DIGITS[digit]}{unit}")

    text = "".join(parts)
    # 习惯写法：十一而不是一十一，但一百一十保留。
    if text.startswith("一十"):
        text = text[1:]
    return text


def _chinese_suffix(suffix: str) -> str:
    """「之3」也要写成「之三」，与索引里的写法对齐。"""

    return (
        _CN_DIGITS[int(suffix)]
        if suffix.isdigit()
        else suffix
    )


def normalize_article_numbers(text: str) -> str:
    """把文本里的「第2条」统一成「第二条」。"""

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        # 已经是中文数字的条号保持原样，只转阿拉伯数字。
        number = (
            to_chinese_article_number(int(raw))
            if raw.isdigit()
            else raw
        )
        suffix = match.group(2)
        if suffix:
            return (
                f"第{number}条之{_chinese_suffix(suffix)}"
            )
        return f"第{number}条"

    return _ARTICLE_REFERENCE_PATTERN.sub(
        replace,
        text.translate(_FULL_WIDTH_DIGITS),
    )


def extract_article_references(
    text: str,
) -> list[str]:
    """按出现顺序抽出全部条号，去重后统一成索引里的写法。"""

    normalized = text.translate(_FULL_WIDTH_DIGITS)
    references: list[str] = []

    for match in _ARTICLE_REFERENCE_PATTERN.finditer(
        normalized
    ):
        raw = match.group(1)
        number = (
            to_chinese_article_number(int(raw))
            if raw.isdigit()
            else raw
        )
        reference = f"第{number}条"
        suffix = match.group(2)
        if suffix:
            reference += f"之{_chinese_suffix(suffix)}"
        if reference not in references:
            references.append(reference)

    return references


def law_short_name(law_name: str) -> str:
    """去掉「中华人民共和国」前缀，得到用户口语里的简称。"""

    stripped = law_name.removeprefix(
        NATIONAL_PREFIX
    ).strip()
    return stripped or law_name


class ArticleAnchorIndex:
    """按 (法律名, 条号) 查法条，供融合后补召回。"""

    def __init__(
        self,
        chunks: list[ArticleChunk],
    ) -> None:
        self._chunks = {
            (chunk.law_name, chunk.article_no): chunk
            for chunk in chunks
        }
        self._laws = tuple(
            sorted({chunk.law_name for chunk in chunks})
        )

    @classmethod
    def load(
        cls,
        articles_path: Path = DEFAULT_ARTICLES_PATH,
    ) -> "ArticleAnchorIndex":
        return cls(load_articles(articles_path))

    @property
    def laws(self) -> tuple[str, ...]:
        return self._laws

    def resolve(
        self,
        query: str,
        *,
        law_name: str | None = None,
        limit: int = DEFAULT_ANCHOR_LIMIT,
    ) -> list[RetrievalResult]:
        """返回查询里点名的那些条文；定位不到就返回空列表。"""

        references = extract_article_references(query)
        if not references:
            return []

        target_law = self._resolve_law(
            query,
            law_name=law_name,
        )
        if target_law is None:
            # 没说清是哪部法，硬补会把「第二条」补成别人家的第二条。
            logger.info(
                "[RAG] anchor skipped: ambiguous law "
                "references=%s",
                references,
            )
            return []

        results: list[RetrievalResult] = []

        for reference in references[:limit]:
            chunk = self._chunks.get(
                (target_law, reference)
            )
            if chunk is None:
                continue
            results.append(
                make_retrieval_result(
                    chunk,
                    score=0.0,
                    source="hybrid",
                    component_scores={"anchor": 1.0},
                )
            )

        if results:
            logger.info(
                "[RAG] anchor injected law=%s articles=%s",
                target_law,
                [item.article_no for item in results],
            )

        return results

    def _resolve_law(
        self,
        query: str,
        *,
        law_name: str | None,
    ) -> str | None:
        """先认调用方的法律过滤条件，再认查询里出现的法律名。"""

        if law_name:
            return law_name

        matches = [
            law
            for law in self._laws
            if law in query
            or law_short_name(law) in query
        ]

        if len(matches) == 1:
            return matches[0]

        return None
