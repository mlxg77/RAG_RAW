from rag_law.generation.citation_validator import (
    CitationValidator,
)
from rag_law.schemas import (
    ArticleChunk,
    BuiltContext,
    ContextEvidence,
    GeneratedAnswer,
    GeneratedCitation,
)


def make_chunk() -> ArticleChunk:
    return ArticleChunk(
        chunk_id="chunk-labor-10",
        law_id="cn-labor-contract-law",
        law_name="中华人民共和国劳动合同法",
        article_no="第十条",
        text=(
            "建立劳动关系，应当订立书面劳动合同。"
            "已建立劳动关系，未同时订立书面劳动合同的，"
            "应当自用工之日起一个月内订立书面劳动合同。"
        ),
        source_file=(
            "data/raw/中华人民共和国劳动合同法.pdf"
        ),
        page=3,
        text_hash="hash-labor-10",
    )


def make_context(
    chunk: ArticleChunk,
) -> BuiltContext:
    evidence_text = (
        "建立劳动关系，应当订立书面劳动合同。"
    )

    evidence = ContextEvidence(
        evidence_id="E001",
        chunk_id=chunk.chunk_id,
        law_id=chunk.law_id,
        law_name=chunk.law_name,
        article_no=chunk.article_no,
        text=evidence_text,
        source_file=chunk.source_file,
        page=chunk.page,
        end_page=chunk.end_page,
        relation="matched",
        is_excerpt=True,
        score=0.95,
    )

    return BuiltContext(
        text=(
            "[E001]\n"
            "法律：《中华人民共和国劳动合同法》\n"
            "条号：第十条\n"
            "正文：\n"
            f"{evidence_text}"
        ),
        evidences=[evidence],
        estimated_tokens=50,
        max_tokens=3000,
        truncated=True,
    )


def make_answer(
    citation: GeneratedCitation,
) -> GeneratedAnswer:
    return GeneratedAnswer(
        summary="建立劳动关系应当订立书面劳动合同。",
        analysis="现有证据规定了书面劳动合同要求。",
        conditions=[
            "需要确认双方是否已经建立劳动关系。"
        ],
        citations=[citation],
        limitations=(
            "这里只说明当前知识库中的一般规定。"
        ),
        follow_up_question=None,
    )


def make_valid_citation() -> GeneratedCitation:
    return GeneratedCitation(
        evidence_id="E001",
        law_name="中华人民共和国劳动合同法",
        article_no="第十条",
        quote=(
            "建立劳动关系，应当订立书面劳动合同。"
        ),
    )


def test_valid_citation_is_mapped_to_chunk() -> None:
    chunk = make_chunk()
    validator = CitationValidator(
        chunks=[chunk]
    )

    result = validator.validate(
        answer=make_answer(
            make_valid_citation()
        ),
        context=make_context(chunk),
    )

    assert result.is_valid is True
    assert result.errors == []
    assert len(result.citations) == 1

    citation = result.citations[0]

    assert citation.evidence_id == "E001"
    assert citation.chunk_id == chunk.chunk_id
    assert citation.law_name == chunk.law_name
    assert citation.article_no == chunk.article_no
    assert citation.page == 3


def test_unknown_evidence_id_is_rejected() -> None:
    chunk = make_chunk()
    validator = CitationValidator(
        chunks=[chunk]
    )

    citation = make_valid_citation().model_copy(
        update={"evidence_id": "E999"}
    )

    result = validator.validate(
        answer=make_answer(citation),
        context=make_context(chunk),
    )

    assert result.is_valid is False
    assert result.citations == []
    assert "不存在的证据编号" in result.errors[0]


def test_wrong_law_name_is_rejected() -> None:
    chunk = make_chunk()
    validator = CitationValidator(
        chunks=[chunk]
    )

    citation = make_valid_citation().model_copy(
        update={
            "law_name": "中华人民共和国劳动法"
        }
    )

    result = validator.validate(
        answer=make_answer(citation),
        context=make_context(chunk),
    )

    assert result.is_valid is False
    assert "法律名称不一致" in result.errors[0]


def test_wrong_article_number_is_rejected() -> None:
    chunk = make_chunk()
    validator = CitationValidator(
        chunks=[chunk]
    )

    citation = make_valid_citation().model_copy(
        update={"article_no": "第十一条"}
    )

    result = validator.validate(
        answer=make_answer(citation),
        context=make_context(chunk),
    )

    assert result.is_valid is False
    assert "条号不一致" in result.errors[0]


def test_rewritten_quote_is_rejected() -> None:
    chunk = make_chunk()
    validator = CitationValidator(
        chunks=[chunk]
    )

    citation = make_valid_citation().model_copy(
        update={
            "quote": "用人单位必须签订劳动合同。"
        }
    )

    result = validator.validate(
        answer=make_answer(citation),
        context=make_context(chunk),
    )

    assert result.is_valid is False
    assert result.citations == []
    assert "引文不在本次" in result.errors[0]


def test_quote_outside_displayed_excerpt_is_rejected() -> None:
    chunk = make_chunk()
    validator = CitationValidator(
        chunks=[chunk]
    )

    citation = make_valid_citation().model_copy(
        update={
            "quote": (
                "应当自用工之日起一个月内"
                "订立书面劳动合同。"
            )
        }
    )

    result = validator.validate(
        answer=make_answer(citation),
        context=make_context(chunk),
    )

    assert result.is_valid is False
    assert "引文不在本次" in result.errors[0]


def test_answer_without_citations_has_no_fake_errors() -> None:
    chunk = make_chunk()
    validator = CitationValidator(
        chunks=[chunk]
    )

    answer = GeneratedAnswer(
        summary="当前证据不足，无法确认。",
        analysis="没有足够证据支持具体法律结论。",
        conditions=[],
        citations=[],
        limitations="当前知识库无法回答该问题。",
        follow_up_question=None,
    )

    result = validator.validate(
        answer=answer,
        context=make_context(chunk),
    )

    assert result.is_valid is True
    assert result.citations == []
    assert result.errors == []