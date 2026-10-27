import time
from typing import Any

from rag_law.generation.answer_graph import (
    build_answer_graph,
    route_after_evidence,
    route_after_generate,
    route_after_initial_validation,
    route_after_repaired_validation,
    route_after_repair,
)
from rag_law.generation.chain import (
    GenerationChains,
    GenerationStreamHandlers,
    GenerationStreamResult,
)
from rag_law.generation.citation_validator import (
    CitationValidator,
)
from rag_law.generation.conversation import (
    ConversationContextBuilder,
)
from rag_law.schemas import (
    ArticleChunk,
    BuiltContext,
    ContextEvidence,
    ConversationTurn,
    GeneratedAnswer,
    GeneratedCitation,
    RetrievalContextResult,
    RetrievalResult,
)


class StubRetrievalPipeline:
    """按固定结果返回，并转发两条检索进度事件。"""

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
        progress_callback=None,
    ) -> RetrievalContextResult:
        self.calls.append((query, law_name))

        if progress_callback is not None:
            progress_callback(
                "retrieval",
                "正在进行混合检索与法条重排…",
            )
            progress_callback(
                "context_building",
                "正在筛选相邻法条并构建证据上下文…",
            )

        return self.output


class StubChain:
    def __init__(
        self,
        responses: list[
            GeneratedAnswer | None | Exception
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
    with_evidence: bool = True,
) -> RetrievalContextResult:
    context_text = ""

    if with_evidence:
        context_text = (
            "[E001]\n"
            f"法律：《{chunk.law_name}》\n"
            f"条号：{chunk.article_no}\n"
            "证据类型：直接命中\n"
            "正文：\n"
            f"{chunk.text}"
        )

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
            evidences=(
                [evidence] if with_evidence else []
            ),
            estimated_tokens=50 if with_evidence else 0,
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


def make_graph(
    *,
    retrieval: RetrievalContextResult | None = None,
    answer_responses: list[
        GeneratedAnswer | None | Exception
    ] | None = None,
    repair_responses: list[
        GeneratedAnswer | None | Exception
    ] | None = None,
    streamer=None,
):
    chunk = make_chunk()
    retrieval_pipeline = StubRetrievalPipeline(
        retrieval
        if retrieval is not None
        else make_retrieval(chunk)
    )
    answer_chain = StubChain(
        answer_responses or []
    )
    repair_chain = StubChain(
        repair_responses or []
    )
    deps = StubDependencies(
        retrieval_pipeline=retrieval_pipeline,
        generation_chains=GenerationChains(
            answer_chain=answer_chain,
            repair_chain=repair_chain,
            streamer=streamer,
        ),
        citation_validator=CitationValidator(
            chunks=[chunk]
        ),
    )
    graph = build_answer_graph(deps)

    return (
        graph,
        deps,
        retrieval_pipeline,
        answer_chain,
        repair_chain,
    )


class StubDependencies:
    """实现图依赖协议的最小替身。"""

    def __init__(
        self,
        *,
        retrieval_pipeline,
        generation_chains,
        citation_validator,
        min_answer_score: float = 0.15,
        query_rewriter=None,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.generation_chains = generation_chains
        self.citation_validator = citation_validator
        self.conversation_context_builder = (
            ConversationContextBuilder(
                recent_rounds=2,
                summary_max_chars=200,
                summarizer=None,
            )
        )
        # 不传改写器时退回历史拼接，与线上关掉开关时一致。
        self.query_rewriter = query_rewriter
        self.min_answer_score = min_answer_score


def make_state(
    *,
    progress_callback=None,
    min_answer_score: float = 0.15,
    stream_handlers=None,
) -> dict[str, object]:
    return {
        "query": "公司没有签劳动合同怎么办？",
        "law_name": None,
        "history": None,
        "min_answer_score": min_answer_score,
        "progress_callback": progress_callback,
        "stream_handlers": stream_handlers,
        "total_started_at": time.perf_counter(),
        "attempts": 0,
        "validation_errors": [],
        "generation_started_at": None,
    }


def test_route_after_evidence_requires_sufficient_evidence() -> (
    None
):
    chunk = make_chunk()

    assert (
        route_after_evidence(
            {
                "retrieval": make_retrieval(chunk),
                "min_answer_score": 0.15,
            }
        )
        == "generate"
    )
    assert (
        route_after_evidence(
            {
                "retrieval": make_retrieval(
                    chunk, score=0.05
                ),
                "min_answer_score": 0.15,
            }
        )
        == "insufficient"
    )
    assert (
        route_after_evidence(
            {
                "retrieval": make_retrieval(
                    chunk, with_evidence=False
                ),
                "min_answer_score": 0.15,
            }
        )
        == "insufficient"
    )


def test_route_after_generate_uses_outlet_marker() -> None:
    assert (
        route_after_generate({}) == "validate_initial"
    )
    assert (
        route_after_generate(
            {"answer": make_answer()}
        )
        == "generation_failed"
    )


def test_route_after_initial_validation_limits_repair() -> (
    None
):
    retrieval = make_retrieval(make_chunk())

    assert (
        route_after_initial_validation(
            {
                "retrieval": retrieval,
                "validation": (
                    CitationValidator(
                        chunks=[make_chunk()]
                    ).validate(
                        answer=make_answer(),
                        context=retrieval.context,
                    )
                ),
                "attempts": 1,
            }
        )
        == "finalize"
    )

    invalid = CitationValidator(
        chunks=[make_chunk()]
    ).validate(
        answer=make_answer(evidence_id="E999"),
        context=retrieval.context,
    )

    assert (
        route_after_initial_validation(
            {
                "retrieval": retrieval,
                "validation": invalid,
                "attempts": 1,
            }
        )
        == "repair"
    )
    assert (
        route_after_initial_validation(
            {
                "retrieval": retrieval,
                "validation": invalid,
                "attempts": 2,
            }
        )
        == "generation_failed"
    )


def test_route_after_repair_and_repaired_validation() -> (
    None
):
    retrieval = make_retrieval(make_chunk())
    validator = CitationValidator(
        chunks=[make_chunk()]
    )

    assert (
        route_after_repair({}) == "validate_repaired"
    )
    assert (
        route_after_repair(
            {"answer": make_answer()}
        )
        == "generation_failed"
    )

    assert (
        route_after_repaired_validation(
            {
                "validation": validator.validate(
                    answer=make_answer(),
                    context=retrieval.context,
                )
            }
        )
        == "finalize"
    )
    assert (
        route_after_repaired_validation(
            {
                "validation": validator.validate(
                    answer=make_answer(
                        evidence_id="E998"
                    ),
                    context=retrieval.context,
                )
            }
        )
        == "generation_failed"
    )


def test_graph_answers_with_valid_citation() -> None:
    (
        graph,
        _,
        retrieval,
        answer_chain,
        repair_chain,
    ) = make_graph(
        answer_responses=[make_answer()]
    )

    events: list[str] = []
    output = graph.invoke(
        make_state(
            progress_callback=(
                lambda stage, message: events.append(
                    stage
                )
            )
        )
    )

    assert output["answer"].status == "answered"
    assert output["attempts"] == 1
    assert output["validation_errors"] == []
    assert len(output["answer"].citations) == 1
    assert (
        output["answer"].citations[0].chunk_id
        == "chunk-labor-10"
    )
    assert len(retrieval.calls) == 1
    assert len(answer_chain.calls) == 1
    assert len(repair_chain.calls) == 0
    assert events == [
        "retrieval",
        "context_building",
        "evidence_check",
        "generation",
        "citation_validation",
    ]


def test_graph_refuses_when_evidence_is_insufficient() -> (
    None
):
    (
        graph,
        _,
        _,
        answer_chain,
        repair_chain,
    ) = make_graph(
        retrieval=make_retrieval(
            make_chunk(), score=0.05
        ),
        answer_responses=[make_answer()],
    )

    events: list[str] = []
    output = graph.invoke(
        make_state(
            progress_callback=(
                lambda stage, message: events.append(
                    stage
                )
            )
        )
    )

    assert (
        output["answer"].status
        == "insufficient_evidence"
    )
    assert output["attempts"] == 0
    assert output["generation_started_at"] is None
    assert len(answer_chain.calls) == 0
    assert len(repair_chain.calls) == 0
    assert events == [
        "retrieval",
        "context_building",
        "evidence_check",
    ]


def test_graph_degrades_on_generation_error() -> None:
    (
        graph,
        _,
        _,
        _,
        repair_chain,
    ) = make_graph(
        answer_responses=[
            RuntimeError("模型不可用")
        ]
    )

    events: list[str] = []
    output = graph.invoke(
        make_state(
            progress_callback=(
                lambda stage, message: events.append(
                    stage
                )
            )
        )
    )

    assert (
        output["answer"].status == "generation_failed"
    )
    assert output["attempts"] == 1
    assert output["validation_errors"] == [
        "首次结构化生成失败："
        "RuntimeError: 模型不可用"
    ]
    assert len(repair_chain.calls) == 0
    assert events == [
        "retrieval",
        "context_building",
        "evidence_check",
        "generation",
    ]


def test_graph_repairs_invalid_citation_once() -> None:
    (
        graph,
        _,
        _,
        _,
        repair_chain,
    ) = make_graph(
        answer_responses=[
            make_answer(evidence_id="E999")
        ],
        repair_responses=[make_answer()],
    )

    events: list[str] = []
    output = graph.invoke(
        make_state(
            progress_callback=(
                lambda stage, message: events.append(
                    stage
                )
            )
        )
    )

    assert output["answer"].status == "answered"
    assert output["attempts"] == 2
    assert len(output["answer"].citations) == 1
    assert len(output["validation_errors"]) == 1
    assert output["validation_errors"][0].startswith(
        "首次生成："
    )
    assert len(repair_chain.calls) == 1
    assert events == [
        "retrieval",
        "context_building",
        "evidence_check",
        "generation",
        "citation_validation",
        "repair",
        "citation_validation",
    ]


def test_graph_degrades_when_repair_fails() -> None:
    (
        graph,
        _,
        _,
        _,
        repair_chain,
    ) = make_graph(
        answer_responses=[
            make_answer(evidence_id="E999")
        ],
        repair_responses=[
            make_answer(evidence_id="E998")
        ],
    )

    output = graph.invoke(make_state())

    assert (
        output["answer"].status == "generation_failed"
    )
    assert output["attempts"] == 2
    assert any(
        error.startswith("首次生成：")
        for error in output["validation_errors"]
    )
    assert any(
        error.startswith("修复生成：")
        for error in output["validation_errors"]
    )
    assert len(output["answer"].citations) == 0
    assert len(repair_chain.calls) == 1


def test_graph_skips_repair_after_empty_retry() -> None:
    (
        graph,
        _,
        _,
        answer_chain,
        repair_chain,
    ) = make_graph(
        answer_responses=[
            None,
            make_answer(evidence_id="E999"),
        ],
        repair_responses=[make_answer()],
    )

    output = graph.invoke(make_state())

    assert (
        output["answer"].status == "generation_failed"
    )
    assert output["attempts"] == 2
    assert len(answer_chain.calls) == 2
    assert len(repair_chain.calls) == 0
    assert any(
        error.startswith("首次生成：")
        for error in output["validation_errors"]
    )


def test_graph_swallows_progress_callback_error() -> (
    None
):
    (
        graph,
        _,
        _,
        _,
        _,
    ) = make_graph(
        answer_responses=[make_answer()]
    )

    def broken_callback(
        stage: str,
        message: str,
    ) -> None:
        raise RuntimeError("界面已断开")

    output = graph.invoke(
        make_state(progress_callback=broken_callback)
    )

    assert output["answer"].status == "answered"


def test_graph_retries_once_on_empty_output() -> None:
    (
        graph,
        _,
        _,
        answer_chain,
        _,
    ) = make_graph(
        answer_responses=[
            None,
            make_answer(),
        ]
    )

    output = graph.invoke(make_state())

    assert output["answer"].status == "answered"
    assert output["attempts"] == 2
    assert len(answer_chain.calls) == 2


class StubStreamer:
    """模拟流式生成通道：推一段思考与一个字段快照。"""

    def __init__(
        self,
        answer: GeneratedAnswer,
        *,
        reasoning: str = "先比对证据再下结论",
    ) -> None:
        self.answer = answer
        self.reasoning = reasoning
        self.calls: list[Any] = []

    def stream(
        self,
        payload: dict[str, Any],
        handlers: GenerationStreamHandlers | None = None,
    ) -> GenerationStreamResult:
        self.calls.append(handlers)

        if handlers is not None:
            if handlers.on_reasoning is not None:
                handlers.on_reasoning(self.reasoning)
            if handlers.on_field is not None:
                handlers.on_field("summary", self.answer.summary)

        return GenerationStreamResult(
            answer=self.answer,
            reasoning=self.reasoning,
        )


def test_graph_prefers_streamer_and_keeps_reasoning() -> (
    None
):
    answer = make_answer()
    streamer = StubStreamer(answer)
    (
        graph,
        _,
        _,
        answer_chain,
        _,
    ) = make_graph(
        answer_responses=[answer],
        streamer=streamer,
    )

    reasoning_events: list[str] = []
    field_events: list[tuple[str, str]] = []

    output = graph.invoke(
        make_state(
            stream_handlers=GenerationStreamHandlers(
                on_reasoning=reasoning_events.append,
                on_field=lambda field, text: field_events.append(
                    (field, text)
                ),
            )
        )
    )

    assert output["answer"].status == "answered"
    assert output["reasoning"] == "先比对证据再下结论"
    assert reasoning_events == ["先比对证据再下结论"]
    assert field_events == [("summary", answer.summary)]

    # 流式通道接管后，一次性生成链不应被调用。
    assert len(answer_chain.calls) == 0
    assert len(streamer.calls) == 1


class StubSummaryCache:
    def __init__(self, summary: str = "", covered_seq: int = 0) -> None:
        self.summary = summary
        self.covered_seq = covered_seq
        self.loads = 0
        self.saved: list[tuple[str, int]] = []

    def load(self) -> tuple[str, int]:
        self.loads += 1
        return self.summary, self.covered_seq

    def save(self, summary: str, covered_seq: int) -> None:
        self.saved.append((summary, covered_seq))


def _three_round_history() -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    for index in range(1, 4):
        start = (index - 1) * 2 + 1
        turns.append(
            ConversationTurn(
                role="user", content=f"第 {index} 轮问题", seq=start
            )
        )
        turns.append(
            ConversationTurn(
                role="assistant",
                content=f"第 {index} 轮回答",
                seq=start + 1,
            )
        )
    return turns


def test_prepare_node_uses_summary_cache_from_state() -> None:
    graph, *_ = make_graph(answer_responses=[make_answer()])

    cached = StubSummaryCache("旧摘要", 2)
    state = make_state()
    state["history"] = _three_round_history()
    state["summary_cache"] = cached
    graph.invoke(state)

    assert cached.loads == 1
    # 早期消息已被摘要覆盖，这一轮不必再调模型，也不写回。
    assert cached.saved == []

    # 这个 builder 没有摘要器，降级成原文摘录时不能占住缓存位。
    plain = StubSummaryCache()
    state = make_state()
    state["history"] = _three_round_history()
    state["summary_cache"] = plain
    graph.invoke(state)

    assert plain.loads == 1
    assert plain.saved == []


class StubQueryRewriter:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[str, str | None]] = []

    def rewrite(
        self,
        question,
        conversation,
        *,
        law_name=None,
    ) -> str:
        self.calls.append((question, law_name))
        return self.result


def test_prepare_node_rewrites_follow_up_query() -> None:
    graph, deps, retrieval_pipeline, *_ = make_graph(
        answer_responses=[make_answer()]
    )
    rewriter = StubQueryRewriter(
        "中华人民共和国刑法 第二条"
    )
    deps.query_rewriter = rewriter

    state = make_state()
    state["query"] = "第二条呢"
    state["history"] = _three_round_history()
    graph.invoke(state)

    # 送去检索的是改写后的一句，而不是三轮问题的拼接。
    assert (
        retrieval_pipeline.calls[0][0]
        == "中华人民共和国刑法 第二条"
    )
    assert rewriter.calls == [("第二条呢", None)]
