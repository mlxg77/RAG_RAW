from typing import Any

from rag_law.generation.answer_pipeline import (
    AnswerPipeline,
)
from rag_law.generation.chain import (
    GenerationChains,
)
from rag_law.generation.citation_validator import (
    CitationValidator,
)
from rag_law.schemas import (
    ArticleChunk,
    BuiltContext,
    ContextEvidence,
    GeneratedAnswer,
    GeneratedCitation,
    RetrievalContextResult,
    RetrievalResult,
)


class StubRetrievalPipeline:
    def __init__(
        self,
        output: RetrievalContextResult,
    ) -> None:
        self.output = output
        self.calls: list[
            tuple[str, str | None]
        ] = []

    def run(
        self,
        query: str,
        *,
        law_name: str | None = None,
    ) -> RetrievalContextResult:
        self.calls.append(
            (query, law_name)
        )
        return self.output


class StubChain:
    def __init__(
        self,
        responses: list[
            GeneratedAnswer | Exception
        ],
    ) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        values: dict[str, Any],
    ) -> GeneratedAnswer:
        self.calls.append(values)

        if not self.responses:
            raise AssertionError(
                "StubChain 没有剩余响应"
            )

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


def make_chunk() -> ArticleChunk:
    return ArticleChunk(
        chunk_id="chunk-labor-10",
        law_id="cn-labor-contract-law",
        law_name="中华人民共和国劳动合同法",
        article_no="第十条",
        text=(
            "建立劳动关系，应当订立书面劳动合同。"
        ),
        source_file=(
            "data/raw/中华人民共和国劳动合同法.pdf"
        ),
        page=3,
        text_hash="hash-labor-10",
    )


def make_retrieval(
    chunk: ArticleChunk,
    *,
    score: float = 0.90,
) -> RetrievalContextResult:
    evidence = ContextEvidence(
        evidence_id="E001",
        chunk_id=chunk.chunk_id,
        law_id=chunk.law_id,
        law_name=chunk.law_name,
        article_no=chunk.article_no,
        text=chunk.text,
        source_file=chunk.source_file,
        page=chunk.page,
        end_page=chunk.end_page,
        relation="matched",
        is_excerpt=False,
        score=score,
    )

    context_text = (
        "[E001]\n"
        f"法律：《{chunk.law_name}》\n"
        f"条号：{chunk.article_no}\n"
        "证据类型：直接命中\n"
        "正文：\n"
        f"{chunk.text}"
    )

    return RetrievalContextResult(
        query="公司没有签劳动合同怎么办？",
        reranked_results=[
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                law_id=chunk.law_id,
                law_name=chunk.law_name,
                article_no=chunk.article_no,
                text=chunk.text,
                source_file=chunk.source_file,
                page=chunk.page,
                end_page=chunk.end_page,
                score=score,
                source="rerank",
            )
        ],
        context=BuiltContext(
            text=context_text,
            evidences=[evidence],
            estimated_tokens=50,
            max_tokens=3000,
            truncated=False,
        ),
        retrieval_latency_ms=1.0,
        context_latency_ms=1.0,
        total_latency_ms=2.0,
    )


def make_answer(
    *,
    evidence_id: str = "E001",
    include_citation: bool = True,
    follow_up_question: str | None = None,
) -> GeneratedAnswer:
    citations = []

    if include_citation:
        citations.append(
            GeneratedCitation(
                evidence_id=evidence_id,
                law_name=(
                    "中华人民共和国劳动合同法"
                ),
                article_no="第十条",
                quote=(
                    "建立劳动关系，应当订立"
                    "书面劳动合同。"
                ),
            )
        )

    return GeneratedAnswer(
        summary=(
            "建立劳动关系应当订立书面劳动合同。"
        ),
        analysis=(
            "证据规定了书面劳动合同要求。"
        ),
        conditions=[
            "需要确认是否已经建立劳动关系。"
        ],
        citations=citations,
        limitations=(
            "回答仅基于当前知识库，"
            "不能替代正式法律意见。"
        ),
        follow_up_question=follow_up_question,
    )


def make_pipeline(
    *,
    score: float = 0.90,
    answer_responses: list[
        GeneratedAnswer | Exception
    ],
    repair_responses: list[
        GeneratedAnswer | Exception
    ] | None = None,
):
    chunk = make_chunk()
    retrieval = StubRetrievalPipeline(
        make_retrieval(
            chunk,
            score=score,
        )
    )
    answer_chain = StubChain(
        answer_responses
    )
    repair_chain = StubChain(
        repair_responses or []
    )

    pipeline = AnswerPipeline(
        retrieval_pipeline=retrieval,
        generation_chains=GenerationChains(
            answer_chain=answer_chain,
            repair_chain=repair_chain,
        ),
        citation_validator=CitationValidator(
            chunks=[chunk]
        ),
        min_answer_score=0.15,
    )

    return (
        pipeline,
        retrieval,
        answer_chain,
        repair_chain,
    )


def test_valid_answer_is_returned() -> None:
    (
        pipeline,
        retrieval,
        answer_chain,
        repair_chain,
    ) = make_pipeline(
        answer_responses=[
            make_answer()
        ]
    )

    result = pipeline.run(
        "  公司没有签劳动合同怎么办？  "
    )

    assert result.query == (
        "公司没有签劳动合同怎么办？"
    )
    assert result.answer.status == "answered"
    assert result.generation_attempts == 1
    assert len(result.answer.citations) == 1
    assert (
        result.answer.citations[0].chunk_id
        == "chunk-labor-10"
    )
    assert result.validation_errors == []

    assert len(retrieval.calls) == 1
    assert len(answer_chain.calls) == 1
    assert len(repair_chain.calls) == 0


def test_low_score_refuses_without_calling_llm() -> None:
    (
        pipeline,
        _,
        answer_chain,
        repair_chain,
    ) = make_pipeline(
        score=0.10,
        answer_responses=[],
    )

    result = pipeline.run(
        "知识库范围外的问题"
    )

    assert (
        result.answer.status
        == "insufficient_evidence"
    )
    assert result.answer.citations == []
    assert result.generation_attempts == 0
    assert len(answer_chain.calls) == 0
    assert len(repair_chain.calls) == 0


def test_invalid_citation_is_repaired_once() -> None:
    (
        pipeline,
        _,
        answer_chain,
        repair_chain,
    ) = make_pipeline(
        answer_responses=[
            make_answer(
                evidence_id="E999"
            )
        ],
        repair_responses=[
            make_answer()
        ],
    )

    result = pipeline.run(
        "公司没有签劳动合同怎么办？"
    )

    assert result.answer.status == "answered"
    assert result.generation_attempts == 2
    assert len(result.answer.citations) == 1
    assert len(result.validation_errors) == 1

    assert len(answer_chain.calls) == 1
    assert len(repair_chain.calls) == 1

    repair_input = repair_chain.calls[0]

    assert "E999" in (
        repair_input["previous_answer"]
    )
    assert "不存在的证据编号" in (
        repair_input["validation_errors"]
    )


def test_two_invalid_answers_degrade_safely() -> None:
    (
        pipeline,
        _,
        _,
        repair_chain,
    ) = make_pipeline(
        answer_responses=[
            make_answer(
                evidence_id="E999"
            )
        ],
        repair_responses=[
            make_answer(
                evidence_id="E998"
            )
        ],
    )

    result = pipeline.run(
        "公司没有签劳动合同怎么办？"
    )

    assert (
        result.answer.status
        == "generation_failed"
    )
    assert result.answer.citations == []
    assert result.generation_attempts == 2
    assert len(result.validation_errors) == 2
    assert len(repair_chain.calls) == 1


def test_structured_output_error_degrades_safely() -> None:
    (
        pipeline,
        _,
        answer_chain,
        repair_chain,
    ) = make_pipeline(
        answer_responses=[
            ValueError(
                "无法解析结构化输出"
            )
        ],
    )

    result = pipeline.run(
        "公司没有签劳动合同怎么办？"
    )

    assert (
        result.answer.status
        == "generation_failed"
    )
    assert result.answer.citations == []
    assert result.generation_attempts == 1
    assert "ValueError" in (
        result.validation_errors[0]
    )

    assert len(answer_chain.calls) == 1
    assert len(repair_chain.calls) == 0


def test_follow_up_question_changes_status() -> None:
    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline(
        answer_responses=[
            make_answer(
                follow_up_question=(
                    "双方是否已经建立劳动关系？"
                )
            )
        ],
    )

    result = pipeline.run(
        "对方应该和我签合同吗？"
    )

    assert (
        result.answer.status
        == "needs_clarification"
    )
    assert len(result.answer.citations) == 1
    assert result.answer.follow_up_question == (
        "双方是否已经建立劳动关系？"
    )


def test_answer_without_citation_is_not_exposed() -> None:
    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline(
        answer_responses=[
            make_answer(
                include_citation=False
            )
        ],
    )

    result = pipeline.run(
        "公司没有签劳动合同怎么办？"
    )

    assert (
        result.answer.status
        == "insufficient_evidence"
    )
    assert result.answer.citations == []
    assert "证据规定了" not in (
        result.answer.analysis
    )