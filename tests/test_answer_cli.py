from types import SimpleNamespace

from rag_law.cli import (
    build_parser,
    format_source_location,
    print_answer_result,
)


def make_output():
    citation = SimpleNamespace(
        evidence_id="E001",
        chunk_id="chunk-labor-10",
        law_name="中华人民共和国劳动合同法",
        article_no="第十条",
        quote=(
            "建立劳动关系，应当订立书面劳动合同。"
        ),
        source_file=(
            "data/raw/中华人民共和国劳动合同法.pdf"
        ),
        page=3,
        end_page=3,
    )

    answer = SimpleNamespace(
        status="answered",
        summary="应当订立书面劳动合同。",
        analysis="检索证据规定了书面合同要求。",
        conditions=[
            "需要确认是否已经建立劳动关系。"
        ],
        citations=[citation],
        limitations="仅基于当前知识库。",
        follow_up_question=None,
    )

    context = SimpleNamespace(
        evidences=[
            SimpleNamespace(
                evidence_id="E001"
            )
        ]
    )

    retrieval = SimpleNamespace(
        retrieval_latency_ms=10.0,
        context_latency_ms=5.0,
        context=context,
    )

    return SimpleNamespace(
        query="公司没有签劳动合同怎么办？",
        answer=answer,
        retrieval=retrieval,
        generation_attempts=1,
        top_relevance_score=0.95,
        validation_errors=[
            "内部错误不应在默认输出中出现"
        ],
        prompt_version="stage4-v1",
        generation_latency_ms=100.0,
        total_latency_ms=115.0,
    )


def test_parser_accepts_answer_command() -> None:
    parser = build_parser()

    arguments = parser.parse_args(
        [
            "answer",
            "--query",
            "公司没有签劳动合同怎么办？",
            "--min-answer-score",
            "0.2",
            "--debug",
        ]
    )

    assert arguments.command == "answer"
    assert arguments.query == (
        "公司没有签劳动合同怎么办？"
    )
    assert arguments.min_answer_score == 0.2
    assert arguments.debug is True
    assert arguments.json is False


def test_source_location_formats_page_range() -> None:
    location = format_source_location(
        source_file="data/raw/test.pdf",
        page=3,
        end_page=5,
    )

    assert location == (
        "data/raw/test.pdf，第 3-5 页"
    )


def test_normal_output_hides_internal_errors(
    capsys,
) -> None:
    print_answer_result(
        make_output(),
        debug=False,
    )

    printed = capsys.readouterr().out

    assert "已回答" in printed
    assert "中华人民共和国劳动合同法" in printed
    assert "第十条" in printed
    assert "chunk-labor-10" in printed
    assert "建立劳动关系" in printed
    assert "内部错误" not in printed


def test_debug_output_shows_diagnostics(
    capsys,
) -> None:
    print_answer_result(
        make_output(),
        debug=True,
    )

    printed = capsys.readouterr().out

    assert "Prompt 版本：stage4-v1" in printed
    assert "检索耗时：10.0 ms" in printed
    assert "上下文证据数：1" in printed
    assert "内部错误" in printed
