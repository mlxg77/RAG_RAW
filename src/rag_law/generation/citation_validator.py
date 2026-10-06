"""校验大模型生成的法条引用。"""

from rag_law.retrieval.common import (
    load_articles,
)
from rag_law.schemas import (
    ArticleChunk,
    BuiltContext,
    CitationValidationResult,
    GeneratedAnswer,
    ValidatedCitation,
)


class CitationValidator:
    """把模型引用映射回原始法条并严格校验。"""

    def __init__(
        self,
        *,
        chunks: list[ArticleChunk],
    ) -> None:
        if not chunks:
            raise ValueError(
                "引用校验所需的法条数据不能为空"
            )

        self.chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in chunks
        }

        if len(self.chunks_by_id) != len(chunks):
            raise ValueError(
                "引用校验法条数据包含重复 chunk_id"
            )

    @classmethod
    def load(cls) -> "CitationValidator":
        """从默认 articles.jsonl 加载原始法条。"""

        return cls(
            chunks=load_articles()
        )

    def validate(
        self,
        *,
        answer: GeneratedAnswer,
        context: BuiltContext,
    ) -> CitationValidationResult:
        """校验回答中的全部引用。

        任何一个引用失败，整个结果都视为失败。
        已经通过的引用仍然保留，便于记录和排查。
        """

        evidences_by_id = {
            evidence.evidence_id: evidence
            for evidence in context.evidences
        }

        errors: list[str] = []
        validated_citations: list[
            ValidatedCitation
        ] = []

        for index, citation in enumerate(
            answer.citations,
            start=1,
        ):
            label = f"第 {index} 个引用"

            evidence = evidences_by_id.get(
                citation.evidence_id
            )

            if evidence is None:
                errors.append(
                    f"{label}使用了不存在的证据编号："
                    f"{citation.evidence_id}"
                )
                continue

            chunk = self.chunks_by_id.get(
                evidence.chunk_id
            )

            if chunk is None:
                errors.append(
                    f"{label}对应的 chunk_id "
                    "无法映射到原始法条："
                    f"{evidence.chunk_id}"
                )
                continue

            if (
                evidence.law_name
                != chunk.law_name
                or evidence.article_no
                != chunk.article_no
            ):
                errors.append(
                    f"{label}的上下文证据元数据"
                    "与原始法条不一致"
                )
                continue

            if evidence.text not in chunk.text:
                errors.append(
                    f"{label}的上下文正文"
                    "不是原始法条的连续子串"
                )
                continue

            # 模型只负责选择 evidence_id。法律名称、条号和引文
            # 均从已经校验过的上下文证据回填，避免模型在复制长文本
            # 时产生微小差异并触发一次昂贵的修复生成。
            quote = evidence.text.strip()

            validated_citations.append(
                ValidatedCitation(
                    evidence_id=(
                        citation.evidence_id
                    ),
                    chunk_id=chunk.chunk_id,
                    law_id=chunk.law_id,
                    law_name=chunk.law_name,
                    article_no=chunk.article_no,
                    quote=quote,
                    source_file=(
                        chunk.source_file
                    ),
                    page=chunk.page,
                    end_page=chunk.end_page,
                )
            )

        return CitationValidationResult(
            is_valid=not errors,
            citations=validated_citations,
            errors=errors,
        )
