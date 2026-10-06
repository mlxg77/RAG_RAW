from pathlib import Path

import pytest

from rag_law.schemas import (
    AnswerPipelineResult,
    AnswerResponse,
    BuiltContext,
    RetrievalContextResult,
    ValidatedCitation,
)
from rag_law.ui.presentation import (
    citation_location,
    from_session_payload,
    load_law_options,
    result_metrics,
    status_presentation,
    to_session_payload,
)


def _citation(**overrides) -> ValidatedCitation:
    values = {
        "evidence_id": "E001",
        "chunk_id": "law:1",
        "law_id": "law",
        "law_name": "中华人民共和国测试法",
        "article_no": "第一条",
        "quote": "这是经过校验的法条原文。",
        "source_file": "data/raw/test.pdf",
        "page": 2,
        "end_page": 3,
    }
    values.update(overrides)
    return ValidatedCitation(**values)


def _result() -> AnswerPipelineResult:
    return AnswerPipelineResult(
        query="测试问题",
        answer=AnswerResponse(
            status="answered",
            summary="简明结论",
            analysis="分析",
            conditions=["条件一"],
            citations=[_citation()],
            limitations="限制",
        ),
        retrieval=RetrievalContextResult(
            query="测试问题",
            reranked_results=[],
            context=BuiltContext(
                text="",
                evidences=[],
                estimated_tokens=0,
                max_tokens=100,
                truncated=False,
            ),
            retrieval_latency_ms=100,
            context_latency_ms=10,
            total_latency_ms=110,
        ),
        generation_attempts=1,
        top_relevance_score=0.9,
        prompt_version="test-v1",
        generation_latency_ms=1000,
        total_latency_ms=2500,
    )


def test_load_law_options_preserves_manifest_order(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "- law_id: first\n  law_name: 第一部法\n"
        "- law_id: second\n  law_name: 第二部法\n",
        encoding="utf-8",
    )

    options = load_law_options(manifest)

    assert [option.law_id for option in options] == ["first", "second"]
    assert options[1].law_name == "第二部法"


def test_load_law_options_rejects_missing_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("- law_id: incomplete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="law_id 或 law_name"):
        load_law_options(manifest)


def test_citation_location_formats_page_range() -> None:
    assert citation_location(_citation()) == "data/raw/test.pdf · 第 2–3 页"
    assert citation_location(_citation(end_page=2)) == "data/raw/test.pdf · 第 2 页"
    assert citation_location(_citation(page=None, end_page=None)) == "data/raw/test.pdf"


def test_status_and_metrics_are_user_facing() -> None:
    result = _result()

    assert status_presentation("answered").label == "已找到依据"
    assert result_metrics(result) == {
        "引用": "1",
        "总耗时": "2.5 秒",
        "生成次数": "1",
    }


def test_session_payload_round_trip() -> None:
    result = _result()

    restored = from_session_payload(to_session_payload(result))

    assert restored == result
