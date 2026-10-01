import json

from rag_law.generation.run_logger import (
    AnswerRunLogger,
    redact_text,
)
from rag_law.schemas import (
    AnswerPipelineResult,
    AnswerResponse,
    BuiltContext,
    RetrievalContextResult,
    RetrievalResult,
)


def make_result() -> AnswerPipelineResult:
    retrieval_item = RetrievalResult(
        chunk_id="chunk-labor-10",
        law_id="cn-labor-contract-law",
        law_name="中华人民共和国劳动合同法",
        article_no="第十条",
        text=(
            "这段法条全文不应重复写入运行日志。"
        ),
        source_file=(
            "data/raw/中华人民共和国劳动合同法.pdf"
        ),
        page=3,
        score=0.95,
        source="rerank",
    )

    retrieval = RetrievalContextResult(
        query=(
            "我的手机号是13812345678，"
            "公司没有签劳动合同怎么办？"
        ),
        reranked_results=[
            retrieval_item
        ],
        context=BuiltContext(
            text=(
                "这段完整模型上下文"
                "不应写入运行日志。"
            ),
            evidences=[],
            estimated_tokens=20,
            max_tokens=3000,
            truncated=False,
        ),
        retrieval_latency_ms=10.0,
        context_latency_ms=5.0,
        total_latency_ms=15.0,
    )

    answer = AnswerResponse(
        status="insufficient_evidence",
        summary="暂时无法确认。",
        analysis=(
            "请不要公开身份证号"
            "11010519491231002X，"
            "联系邮箱为test@example.com。"
        ),
        conditions=[],
        citations=[],
        limitations="仅基于当前知识库。",
        follow_up_question=None,
    )

    return AnswerPipelineResult(
        query=retrieval.query,
        answer=answer,
        retrieval=retrieval,
        generation_attempts=0,
        top_relevance_score=0.95,
        validation_errors=[
            "这段内部错误不应写入日志"
        ],
        prompt_version="stage4-v1",
        generation_latency_ms=0.0,
        total_latency_ms=15.0,
    )


def test_redact_text_removes_sensitive_values() -> None:
    original = (
        "身份证11010519491231002X，"
        "手机13812345678，"
        "邮箱test@example.com，"
        "银行卡6222 0202 0123 4567。"
    )

    redacted, detected_types = (
        redact_text(original)
    )

    assert "11010519491231002X" not in redacted
    assert "13812345678" not in redacted
    assert "test@example.com" not in redacted
    assert "6222 0202 0123 4567" not in redacted

    assert detected_types == {
        "id_card",
        "mobile_phone",
        "email",
        "bank_card",
    }


def test_logger_writes_one_redacted_json_line(
    tmp_path,
) -> None:
    log_path = tmp_path / "answer_runs.jsonl"

    logger = AnswerRunLogger(
        path=log_path,
        model_name="test-model",
    )

    logger.write(
        result=make_result(),
        law_name="中华人民共和国劳动合同法",
    )

    lines = log_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["schema_version"] == "1.0"
    assert record["model_name"] == "test-model"
    assert record["prompt_version"] == "stage4-v1"

    assert (
        record["status"]
        == "insufficient_evidence"
    )

    assert "[手机号已脱敏]" in record["query"]
    assert (
        "[身份证号已脱敏]"
        in record["answer"]["analysis"]
    )
    assert (
        "[邮箱已脱敏]"
        in record["answer"]["analysis"]
    )

    assert record["redaction_types"] == [
        "email",
        "id_card",
        "mobile_phone",
    ]

    assert (
        record["metrics"][
            "validation_error_count"
        ]
        == 1
    )


def test_logger_does_not_store_raw_context_or_errors(
    tmp_path,
) -> None:
    log_path = tmp_path / "answer_runs.jsonl"

    logger = AnswerRunLogger(
        path=log_path,
        model_name="test-model",
    )

    logger.write(
        result=make_result()
    )

    raw_log = log_path.read_text(
        encoding="utf-8"
    )

    assert "13812345678" not in raw_log
    assert "11010519491231002X" not in raw_log
    assert "test@example.com" not in raw_log

    assert "完整模型上下文" not in raw_log
    assert "法条全文不应重复" not in raw_log
    assert "内部错误不应写入日志" not in raw_log

    record = json.loads(
        raw_log.strip()
    )

    retrieval_item = (
        record["retrieval_results"][0]
    )

    assert retrieval_item["chunk_id"] == (
        "chunk-labor-10"
    )
    assert "text" not in retrieval_item