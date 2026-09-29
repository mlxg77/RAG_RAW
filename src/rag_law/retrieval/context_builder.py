"""将精排结果构造成可追溯、去重且长度受控的上下文。"""

import re
from dataclasses import dataclass

from rag_law.retrieval.common import (
    load_articles,
    make_retrieval_result,
)
from rag_law.retrieval.reranker import (
    RerankScorerProtocol,
    build_rerank_text,
)
from rag_law.schemas import (
    ArticleChunk,
    BuiltContext,
    ContextEvidence,
    ContextRelation,
    RetrievalResult,
)


DEFAULT_MAX_CONTEXT_TOKENS = 3000
DEFAULT_MAX_EVIDENCES = 6
DEFAULT_MAX_EVIDENCE_TOKENS = 700
DEFAULT_ADJACENT_FOR_TOP_N = 2
DEFAULT_MIN_ADJACENT_SCORE = 0.35

# 只在完整句号、问号、叹号、分号或自然换行处切分。
# 不使用固定字符长度切分。
UNIT_BOUNDARY_PATTERN = re.compile(
    r"[。！？；]|\r?\n+"
)


@dataclass(frozen=True)
class _ContextCandidate:
    """上下文构建过程中的内部候选。"""

    result: RetrievalResult
    relation: ContextRelation


def estimate_tokens(text: str) -> int:
    """保守估算中文上下文所占 token。

    当前项目使用远程模型，无法直接取得与服务端完全一致的
    tokenizer。中文法律文本中，一个非空白字符按一个 token
    计算，通常偏保守，不容易超过真实上下文窗口。

    阶段 4 如果模型供应商提供确定 tokenizer，可以替换此函数，
    ContextBuilder 的其他逻辑不需要变化。
    """

    return len(
        re.sub(
            r"\s+",
            "",
            text,
        )
    )


def normalize_body(text: str) -> str:
    """生成正文去重键。

    去掉换行、空格和制表符，但不修改最终展示的原始正文。
    """

    return re.sub(
        r"\s+",
        "",
        text,
    )


def split_text_spans(
    text: str,
) -> list[tuple[int, int]]:
    """在句子或自然段边界处分割，并保留原文字符位置。

    返回的每个元素都是 [start, end)。
    使用位置而不是重新拼接句子，可以保证最终片段是原文的
    连续子串。
    """

    spans: list[tuple[int, int]] = []
    start = 0

    def append_span(
        left: int,
        right: int,
    ) -> None:
        raw_text = text[left:right]

        leading_length = (
            len(raw_text)
            - len(raw_text.lstrip())
        )
        trailing_length = (
            len(raw_text)
            - len(raw_text.rstrip())
        )

        actual_left = left + leading_length
        actual_right = right - trailing_length

        if actual_left < actual_right:
            spans.append(
                (
                    actual_left,
                    actual_right,
                )
            )

    for match in (
        UNIT_BOUNDARY_PATTERN.finditer(text)
    ):
        end = match.end()
        append_span(start, end)
        start = end

    if start < len(text):
        append_span(
            start,
            len(text),
        )

    # 没有任何标点但正文非空时，把整段看成一个不可再分单位。
    if not spans and text.strip():
        left = len(text) - len(
            text.lstrip()
        )
        right = len(text.rstrip())

        spans.append((left, right))

    return spans


def render_evidence(
    evidence: ContextEvidence,
) -> str:
    """将结构化证据渲染成提供给模型的文本。"""

    relation_text = {
        "matched": "直接命中",
        "adjacent": "相邻条文",
    }[evidence.relation]

    lines = [
        f"[{evidence.evidence_id}]",
        (
            f"法律：《{evidence.law_name}》"
        ),
        f"条号：{evidence.article_no}",
        f"证据类型：{relation_text}",
    ]

    if evidence.chapter:
        lines.append(
            f"章节：{evidence.chapter}"
        )

    lines.extend(
        [
            "正文：",
            evidence.text,
        ]
    )

    return "\n".join(lines)


class ContextBuilder:
    """从 Rerank 结果构建模型上下文。"""

    def __init__(
        self,
        *,
        chunks: list[ArticleChunk],
        scorer: RerankScorerProtocol,
        max_context_tokens: int = (
            DEFAULT_MAX_CONTEXT_TOKENS
        ),
        max_evidences: int = (
            DEFAULT_MAX_EVIDENCES
        ),
        max_evidence_tokens: int = (
            DEFAULT_MAX_EVIDENCE_TOKENS
        ),
        adjacent_for_top_n: int = (
            DEFAULT_ADJACENT_FOR_TOP_N
        ),
        min_adjacent_score: float = (
            DEFAULT_MIN_ADJACENT_SCORE
        ),
    ) -> None:
        if not chunks:
            raise ValueError(
                "上下文法条数据不能为空"
            )

        if max_context_tokens <= 0:
            raise ValueError(
                "max_context_tokens 必须大于 0"
            )

        if max_evidences <= 0:
            raise ValueError(
                "max_evidences 必须大于 0"
            )

        if max_evidence_tokens <= 0:
            raise ValueError(
                "max_evidence_tokens 必须大于 0"
            )

        if adjacent_for_top_n < 0:
            raise ValueError(
                "adjacent_for_top_n 不能小于 0"
            )

        if not 0.0 <= min_adjacent_score <= 1.0:
            raise ValueError(
                "min_adjacent_score 必须在 [0, 1] 范围内"
            )

        self.chunks = chunks
        self.scorer = scorer
        self.max_context_tokens = (
            max_context_tokens
        )
        self.max_evidences = max_evidences
        self.max_evidence_tokens = (
            max_evidence_tokens
        )
        self.adjacent_for_top_n = (
            adjacent_for_top_n
        )
        self.min_adjacent_score = (
            min_adjacent_score
        )

        self.chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in chunks
        }

        self.positions_by_id = {
            chunk.chunk_id: index
            for index, chunk in enumerate(
                chunks
            )
        }

        if len(self.chunks_by_id) != len(
            chunks
        ):
            raise ValueError(
                "上下文法条数据包含重复 chunk_id"
            )

    @classmethod
    def load(
        cls,
        *,
        scorer: RerankScorerProtocol,
        max_context_tokens: int = (
            DEFAULT_MAX_CONTEXT_TOKENS
        ),
        max_evidences: int = (
            DEFAULT_MAX_EVIDENCES
        ),
        max_evidence_tokens: int = (
            DEFAULT_MAX_EVIDENCE_TOKENS
        ),
        adjacent_for_top_n: int = (
            DEFAULT_ADJACENT_FOR_TOP_N
        ),
        min_adjacent_score: float = (
            DEFAULT_MIN_ADJACENT_SCORE
        ),
    ) -> "ContextBuilder":
        """加载默认 articles.jsonl 创建构建器。"""

        return cls(
            chunks=load_articles(),
            scorer=scorer,
            max_context_tokens=(
                max_context_tokens
            ),
            max_evidences=max_evidences,
            max_evidence_tokens=(
                max_evidence_tokens
            ),
            adjacent_for_top_n=(
                adjacent_for_top_n
            ),
            min_adjacent_score=(
                min_adjacent_score
            ),
        )

    @staticmethod
    def _article_key(
        chunk: ArticleChunk,
    ) -> tuple[str, str]:
        return (
            chunk.law_id,
            chunk.article_no,
        )

    def _validate_result(
        self,
        result: RetrievalResult,
    ) -> ArticleChunk:
        """确认检索结果确实映射到原始法条。"""

        chunk = self.chunks_by_id.get(
            result.chunk_id
        )

        if chunk is None:
            raise ValueError(
                "检索结果无法映射到原始法条："
                f"{result.chunk_id}"
            )

        mismatched_fields = [
            field_name
            for field_name in (
                "law_id",
                "law_name",
                "article_no",
                "text",
                "source_file",
            )
            if (
                getattr(chunk, field_name)
                != getattr(result, field_name)
            )
        ]

        if mismatched_fields:
            raise ValueError(
                "检索结果与原始法条内容不一致："
                f"{result.chunk_id}，"
                f"冲突字段：{mismatched_fields}"
            )

        return chunk

    def _find_neighbor(
        self,
        chunk: ArticleChunk,
        *,
        step: int,
    ) -> ArticleChunk | None:
        """查找同一部法律中的上一条或下一条。

        如果以后同一法条拆成多个 paragraph chunk，这里会跳过
        同条的其他 chunk，寻找真正不同的相邻条文。
        """

        if step not in (-1, 1):
            raise ValueError(
                "step 只能是 -1 或 1"
            )

        start_position = (
            self.positions_by_id[
                chunk.chunk_id
            ]
        )

        origin_article_key = (
            self._article_key(chunk)
        )

        position = start_position + step

        while (
            0
            <= position
            < len(self.chunks)
        ):
            neighbor = self.chunks[
                position
            ]

            # articles.jsonl 按法律顺序排列；
            # 遇到下一部法律后停止，绝不跨法律补条。
            if neighbor.law_id != chunk.law_id:
                return None

            if (
                self._article_key(neighbor)
                != origin_article_key
            ):
                return neighbor

            position += step

        return None

    def _build_adjacent_candidates(
        self,
        *,
        query: str,
        matched_chunks: list[ArticleChunk],
    ) -> list[_ContextCandidate]:
        """收集并重排相邻法条。

        相邻条文不能无条件塞进上下文。先收集前后条，再使用同一
        Reranker 对它们打分，相关性较高的邻条优先使用。
        """

        if self.adjacent_for_top_n == 0:
            return []

        neighbor_chunks: dict[
            str,
            ArticleChunk,
        ] = {}

        for chunk in matched_chunks[
            :self.adjacent_for_top_n
        ]:
            for step in (-1, 1):
                neighbor = (
                    self._find_neighbor(
                        chunk,
                        step=step,
                    )
                )

                if neighbor is not None:
                    neighbor_chunks.setdefault(
                        neighbor.chunk_id,
                        neighbor,
                    )

        if not neighbor_chunks:
            return []

        temporary_results = [
            make_retrieval_result(
                chunk,
                score=0.0,
                source="rerank",
                component_scores={
                    "adjacent": 1.0,
                },
            )
            for chunk in neighbor_chunks.values()
        ]

        documents = [
            build_rerank_text(result)
            for result in temporary_results
        ]

        scores = self.scorer.score_documents(
            query,
            documents,
        )

        if len(scores) != len(
            temporary_results
        ):
            raise ValueError(
                "相邻法条重排分数数量不一致"
            )

        scored_results: list[
            RetrievalResult
        ] = []

        for result, score in zip(
            temporary_results,
            scores,
            strict=True,
        ):
            component_scores = dict(
                result.component_scores
            )
            component_scores[
                "adjacent_rerank"
            ] = float(score)

            scored_results.append(
                result.model_copy(
                    update={
                        "score": float(score),
                        "component_scores": (
                            component_scores
                        ),
                    }
                )
            )

        scored_results.sort(
            key=lambda result: (
                -result.score,
                result.chunk_id,
            )
        )

        scored_results = [
            result
            for result in scored_results
            if (
                result.score
                >= self.min_adjacent_score
            )
        ]

        return [
            _ContextCandidate(
                result=result,
                relation="adjacent",
            )
            for result in scored_results
        ]

    def _select_excerpt(
        self,
        *,
        query: str,
        text: str,
    ) -> tuple[str | None, bool]:
        """从超长法条中选取连续且完整的相关片段。

        返回：
        - 片段文本；没有任何完整句能放入预算时返回 None；
        - 是否进行了片段提取。
        """

        stripped_text = text.strip()

        if (
            estimate_tokens(stripped_text)
            <= self.max_evidence_tokens
        ):
            return stripped_text, False

        spans = split_text_spans(text)

        if not spans:
            return None, True

        units = [
            text[start:end]
            for start, end in spans
        ]

        scores = self.scorer.score_documents(
            query,
            units,
        )

        if len(scores) != len(units):
            raise ValueError(
                "法条片段重排分数数量不一致"
            )

        # 先寻找相关性最高且能完整放入预算的句子/自然段。
        ranked_indices = sorted(
            range(len(spans)),
            key=lambda index: (
                -scores[index],
                index,
            ),
        )

        anchor_index: int | None = None

        for index in ranked_indices:
            if (
                estimate_tokens(units[index])
                <= self.max_evidence_tokens
            ):
                anchor_index = index
                break

        # 单个完整句已经超过预算时，不从句子中间截断。
        if anchor_index is None:
            return None, True

        left_index = anchor_index
        right_index = anchor_index

        while True:
            expansion_options: list[
                tuple[
                    float,
                    int,
                    int,
                ]
            ] = []

            if left_index > 0:
                expansion_options.append(
                    (
                        scores[left_index - 1],
                        left_index - 1,
                        right_index,
                    )
                )

            if right_index + 1 < len(spans):
                expansion_options.append(
                    (
                        scores[right_index + 1],
                        left_index,
                        right_index + 1,
                    )
                )

            if not expansion_options:
                break

            expansion_options.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                    item[2],
                )
            )

            expanded = False

            for (
                _,
                new_left_index,
                new_right_index,
            ) in expansion_options:
                excerpt_start = spans[
                    new_left_index
                ][0]
                excerpt_end = spans[
                    new_right_index
                ][1]

                candidate_excerpt = text[
                    excerpt_start:excerpt_end
                ].strip()

                if (
                    estimate_tokens(
                        candidate_excerpt
                    )
                    <= self.max_evidence_tokens
                ):
                    left_index = (
                        new_left_index
                    )
                    right_index = (
                        new_right_index
                    )
                    expanded = True
                    break

            if not expanded:
                break

        excerpt_start = spans[left_index][0]
        excerpt_end = spans[right_index][1]

        excerpt = text[
            excerpt_start:excerpt_end
        ].strip()

        # excerpt 是原文的连续切片，不是多个离散句子的拼接。
        if excerpt not in text:
            raise AssertionError(
                "提取片段必须是原始正文的连续子串"
            )

        return excerpt, True

    def build(
        self,
        *,
        query: str,
        results: list[RetrievalResult],
    ) -> BuiltContext:
        """构建最终上下文。"""

        if not query.strip():
            raise ValueError(
                "上下文问题不能为空"
            )

        if not results:
            return BuiltContext(
                text="",
                evidences=[],
                estimated_tokens=0,
                max_tokens=(
                    self.max_context_tokens
                ),
                truncated=False,
            )

        matched_chunks = [
            self._validate_result(result)
            for result in results
        ]

        candidates = [
            _ContextCandidate(
                result=result,
                relation="matched",
            )
            for result in results
        ]

        candidates.extend(
            self._build_adjacent_candidates(
                query=query,
                matched_chunks=matched_chunks,
            )
        )

        seen_articles: set[
            tuple[str, str]
        ] = set()

        seen_bodies: set[str] = set()

        evidences: list[
            ContextEvidence
        ] = []

        rendered_evidences: list[str] = []

        truncated = False

        for candidate_index, candidate in enumerate(
            candidates
        ):
            if len(evidences) >= self.max_evidences:
                truncated = True
                break

            chunk = self._validate_result(
                candidate.result
            )

            article_key = self._article_key(
                chunk
            )

            body_key = normalize_body(
                chunk.text
            )

            # 两种去重彼此独立：
            # 1. 同一法律、同一条号；
            # 2. 不同记录但正文完全相同。
            if (
                article_key in seen_articles
                or body_key in seen_bodies
            ):
                continue

            excerpt, is_excerpt = (
                self._select_excerpt(
                    query=query,
                    text=chunk.text,
                )
            )

            if excerpt is None:
                # 完整句都无法放进单证据预算，不进行字符级硬切。
                truncated = True
                continue

            evidence_id = (
                f"E{len(evidences) + 1:03d}"
            )

            evidence = ContextEvidence(
                evidence_id=evidence_id,
                chunk_id=chunk.chunk_id,
                law_id=chunk.law_id,
                law_name=chunk.law_name,
                article_no=chunk.article_no,
                text=excerpt,
                part=chunk.part,
                chapter=chunk.chapter,
                section=chunk.section,
                source_file=chunk.source_file,
                page=chunk.page,
                end_page=chunk.end_page,
                relation=candidate.relation,
                is_excerpt=is_excerpt,
                score=candidate.result.score,
            )

            rendered = render_evidence(
                evidence
            )

            proposed_text = "\n\n".join(
                [
                    *rendered_evidences,
                    rendered,
                ]
            )

            if (
                estimate_tokens(proposed_text)
                > self.max_context_tokens
            ):
                # 整段放不下就跳过，不截取半句话。
                truncated = True
                continue

            evidences.append(evidence)
            rendered_evidences.append(
                rendered
            )

            seen_articles.add(article_key)
            seen_bodies.add(body_key)

            if is_excerpt:
                truncated = True

            if (
                len(evidences)
                >= self.max_evidences
                and candidate_index
                < len(candidates) - 1
            ):
                truncated = True

        context_text = "\n\n".join(
            rendered_evidences
        )

        estimated_tokens = estimate_tokens(
            context_text
        )

        if (
            estimated_tokens
            > self.max_context_tokens
        ):
            raise AssertionError(
                "最终上下文超过 token 预算"
            )

        return BuiltContext(
            text=context_text,
            evidences=evidences,
            estimated_tokens=(
                estimated_tokens
            ),
            max_tokens=(
                self.max_context_tokens
            ),
            truncated=truncated,
        )
