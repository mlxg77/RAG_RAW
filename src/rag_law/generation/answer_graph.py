"""基于 LangGraph 的答案生成编排图。

流程：会话准备 -> 检索 -> 证据闸门 -> 结构化生成 -> 引用校验
-> 一次安全修复 -> 安全出口。图只负责编排，生成链、校验器
和运行日志仍由现有组件提供。
"""

import logging
import time
from collections.abc import Callable, Sequence
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_law.generation.chain import (
    GenerationChains,
)
from rag_law.generation.conversation import (
    ConversationContextBuilder,
    ConversationSummaryCacheProtocol,
    PreparedConversation,
    build_contextualized_retrieval_query,
)
from rag_law.schemas import (
    AnswerResponse,
    BuiltContext,
    CitationValidationResult,
    ConversationTurn,
    GeneratedAnswer,
    RetrievalContextResult,
)


logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]


STANDARD_LIMITATION = (
    "本回答仅依据当前知识库中的检索证据，"
    "不能替代律师针对具体事实提供的正式法律意见。"
)


class RetrievalPipelineProtocol(Protocol):
    """生成图依赖的检索门面。"""

    def run(
        self,
        query: str,
        *,
        law_name: str | None = None,
        progress_callback: ProgressCallback
        | None = None,
    ) -> RetrievalContextResult:
        ...


class CitationValidatorProtocol(Protocol):
    """生成图依赖的引用校验接口。"""

    def validate(
        self,
        *,
        answer: GeneratedAnswer,
        context: BuiltContext,
    ) -> CitationValidationResult:
        ...


class AnswerGraphDependencies(Protocol):
    """图节点运行时读取的依赖集合。"""

    retrieval_pipeline: RetrievalPipelineProtocol
    generation_chains: GenerationChains
    citation_validator: CitationValidatorProtocol
    conversation_context_builder: (
        ConversationContextBuilder
    )
    min_answer_score: float


class AnswerGraphState(TypedDict, total=False):
    """生成图在一次请求中流转的全部状态。"""

    query: str
    law_name: str | None
    history: Sequence[ConversationTurn] | None
    # 早期对话摘要缓存；不传则每次请求现算。
    summary_cache: ConversationSummaryCacheProtocol | None
    min_answer_score: float
    progress_callback: ProgressCallback | None
    total_started_at: float

    conversation: PreparedConversation
    retrieval_query: str
    retrieval: RetrievalContextResult
    generated: GeneratedAnswer
    validation: CitationValidationResult
    attempts: int
    validation_errors: list[str]
    generation_started_at: float | None
    answer: AnswerResponse


def emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
) -> None:
    """通知界面当前阶段；展示失败不能中断回答流水线。"""

    logger.info(
        "[RAG] stage=%s message=%s",
        stage,
        message,
    )

    if callback is None:
        return

    try:
        callback(stage, message)
    except Exception:
        logger.exception(
            "[RAG] progress callback failed at stage=%s",
            stage,
        )


def top_score(
    retrieval: RetrievalContextResult,
) -> float | None:
    if not retrieval.reranked_results:
        return None

    return retrieval.reranked_results[0].score


def has_sufficient_evidence(
    retrieval: RetrievalContextResult,
    min_answer_score: float,
) -> bool:
    """调用大模型之前先确认检索证据足够可靠。"""

    relevant_score = top_score(retrieval)

    return (
        relevant_score is not None
        and relevant_score >= min_answer_score
        and bool(retrieval.context.evidences)
        and bool(retrieval.context.text.strip())
    )


def normalize_follow_up(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    normalized = value.strip()

    return normalized or None


def normalize_conditions(
    conditions: list[str],
) -> list[str]:
    return [
        condition.strip()
        for condition in conditions
        if condition.strip()
    ]


def insufficient_answer() -> AnswerResponse:
    return AnswerResponse(
        status="insufficient_evidence",
        summary=(
            "现有知识库未找到足够相关的"
            "法律依据，暂时无法确认。"
        ),
        analysis=(
            "检索结果的相关度不足，"
            "因此系统没有调用模型生成法律结论。"
        ),
        conditions=[],
        citations=[],
        limitations=STANDARD_LIMITATION,
        follow_up_question=None,
    )


def generation_failed_answer() -> AnswerResponse:
    return AnswerResponse(
        status="generation_failed",
        summary=(
            "本次回答未能通过结构化输出"
            "或引用校验，暂时无法提供法律结论。"
        ),
        analysis=(
            "为避免展示无法核验的法律名称、"
            "条号或引文，系统已安全降级。"
        ),
        conditions=[],
        citations=[],
        limitations=STANDARD_LIMITATION,
        follow_up_question=None,
    )


def clarification_answer(
    generated: GeneratedAnswer,
) -> AnswerResponse:
    """没有有效引用但存在追问时，只返回安全追问。"""

    follow_up = normalize_follow_up(
        generated.follow_up_question
    )

    return AnswerResponse(
        status="needs_clarification",
        summary=(
            "还需要确认关键事实后才能判断。"
        ),
        analysis=(
            "当前信息不足以确定哪些法律规定"
            "适用于你的具体情况。"
        ),
        conditions=normalize_conditions(
            generated.conditions
        ),
        citations=[],
        limitations=STANDARD_LIMITATION,
        follow_up_question=follow_up,
    )


def finalize_generated_answer(
    *,
    generated: GeneratedAnswer,
    validation: CitationValidationResult,
) -> AnswerResponse:
    """把模型回答转换成最终可展示回答。"""

    follow_up = normalize_follow_up(
        generated.follow_up_question
    )

    # 没有引用时，不能展示模型生成的实质性法律分析。
    if not validation.citations:
        if follow_up is not None:
            return clarification_answer(generated)

        return insufficient_answer()

    status = (
        "needs_clarification"
        if follow_up is not None
        else "answered"
    )

    return AnswerResponse(
        status=status,
        summary=generated.summary.strip(),
        analysis=generated.analysis.strip(),
        conditions=normalize_conditions(
            generated.conditions
        ),
        citations=validation.citations,
        limitations=(
            generated.limitations.strip()
            or STANDARD_LIMITATION
        ),
        follow_up_question=follow_up,
    )


def format_exception(
    *,
    stage: str,
    error: Exception,
) -> str:
    """生成内部诊断信息，不直接作为用户回答展示。"""

    return (
        f"{stage}失败："
        f"{type(error).__name__}: {error}"
    )


def route_after_evidence(
    state: AnswerGraphState,
) -> str:
    retrieval = state["retrieval"]

    if has_sufficient_evidence(
        retrieval,
        state["min_answer_score"],
    ):
        return "generate"

    return "insufficient"


def route_after_generate(
    state: AnswerGraphState,
) -> str:
    if state.get("answer") is not None:
        return "generation_failed"

    return "validate_initial"


def route_after_initial_validation(
    state: AnswerGraphState,
) -> str:
    validation = state["validation"]

    if validation.is_valid:
        return "finalize"

    # 空结果重试已经用完两次生成机会，不再修复。
    if state.get("attempts", 0) >= 2:
        return "generation_failed"

    return "repair"


def route_after_repair(
    state: AnswerGraphState,
) -> str:
    if state.get("answer") is not None:
        return "generation_failed"

    return "validate_repaired"


def route_after_repaired_validation(
    state: AnswerGraphState,
) -> str:
    if state["validation"].is_valid:
        return "finalize"

    return "generation_failed"


def build_answer_graph(
    deps: AnswerGraphDependencies,
):
    """编译完整的答案生成图。"""

    def prepare(
        state: AnswerGraphState,
    ) -> dict[str, object]:
        conversation = (
            deps.conversation_context_builder.prepare(
                state.get("history"),
                summary_cache=state.get("summary_cache"),
            )
        )

        return {
            "conversation": conversation,
            "retrieval_query": (
                build_contextualized_retrieval_query(
                    state["query"],
                    conversation,
                )
            ),
        }

    def retrieve(
        state: AnswerGraphState,
    ) -> dict[str, object]:
        callback = state.get("progress_callback")
        forwarded = (
            (
                lambda stage, message: emit_progress(
                    callback,
                    stage,
                    message,
                )
            )
            if callback is not None
            else None
        )

        retrieval = deps.retrieval_pipeline.run(
            state["retrieval_query"],
            law_name=state.get("law_name"),
            progress_callback=forwarded,
        )

        logger.info(
            "[RAG] retrieval completed results=%d "
            "retrieval_ms=%.1f context_ms=%.1f",
            len(retrieval.reranked_results),
            retrieval.retrieval_latency_ms,
            retrieval.context_latency_ms,
        )

        return {"retrieval": retrieval}

    def check_evidence(
        state: AnswerGraphState,
    ) -> dict[str, object]:
        emit_progress(
            state.get("progress_callback"),
            "evidence_check",
            "正在评估证据充分性…",
        )

        return {}

    def generate(
        state: AnswerGraphState,
    ) -> dict[str, object]:
        callback = state.get("progress_callback")
        attempts = state.get("attempts", 0) + 1
        started_at = time.perf_counter()
        errors = list(
            state.get("validation_errors", [])
        )

        emit_progress(
            callback,
            "generation",
            "正在依据检索证据生成回答…",
        )

        payload = {
            "question": state["query"],
            "conversation_context": (
                state["conversation"].prompt_text
            ),
            "context": (
                state["retrieval"].context.text
            ),
        }

        attempt_started_at = time.perf_counter()

        try:
            generated = (
                deps.generation_chains
                .answer_chain
                .invoke(payload)
            )

            # 部分 OpenAI 兼容服务偶尔返回空的 tool-call 结果。
            # 这种瞬时空结果允许原样重试一次。
            if generated is None:
                logger.warning(
                    "[RAG] generation attempt=1 "
                    "returned None; retrying once"
                )
                attempts += 1
                generated = (
                    deps.generation_chains
                    .answer_chain
                    .invoke(payload)
                )

            if not isinstance(
                generated,
                GeneratedAnswer,
            ):
                raise TypeError(
                    "结构化生成结果不是 GeneratedAnswer："
                    f"{type(generated)!r}"
                )

            logger.info(
                "[RAG] generation attempt=1 completed "
                "elapsed_ms=%.1f citations=%d",
                (
                    time.perf_counter()
                    - attempt_started_at
                )
                * 1000.0,
                len(generated.citations),
            )

        except Exception as error:
            logger.exception(
                "[RAG] generation attempt=1 failed"
            )
            errors.append(
                format_exception(
                    stage="首次结构化生成",
                    error=error,
                )
            )

            return {
                "attempts": attempts,
                "generation_started_at": started_at,
                "validation_errors": errors,
                "answer": generation_failed_answer(),
            }

        return {
            "generated": generated,
            "attempts": attempts,
            "generation_started_at": started_at,
            "validation_errors": errors,
        }

    def validate_initial(
        state: AnswerGraphState,
    ) -> dict[str, object]:
        emit_progress(
            state.get("progress_callback"),
            "citation_validation",
            "正在校验引用与原始法条…",
        )

        validation = (
            deps.citation_validator.validate(
                answer=state["generated"],
                context=state["retrieval"].context,
            )
        )

        logger.info(
            "[RAG] citation validation completed "
            "attempt=1 valid=%s citations=%d errors=%d",
            validation.is_valid,
            len(validation.citations),
            len(validation.errors),
        )

        if validation.is_valid:
            return {"validation": validation}

        logger.warning(
            "[RAG] citation validation failed "
            "attempt=1 errors=%s",
            validation.errors,
        )

        errors = list(
            state.get("validation_errors", [])
        )
        errors.extend(
            f"首次生成：{error}"
            for error in validation.errors
        )

        return {
            "validation": validation,
            "validation_errors": errors,
        }

    def repair(
        state: AnswerGraphState,
    ) -> dict[str, object]:
        callback = state.get("progress_callback")
        attempts = state.get("attempts", 0) + 1
        errors = list(
            state.get("validation_errors", [])
        )
        validation = state["validation"]

        emit_progress(
            callback,
            "repair",
            "证据编号未通过校验，"
            "正在进行一次安全修复…",
        )

        started_at = time.perf_counter()

        try:
            repaired = (
                deps.generation_chains
                .repair_chain
                .invoke(
                    {
                        "question": state["query"],
                        "conversation_context": (
                            state[
                                "conversation"
                            ].prompt_text
                        ),
                        "context": (
                            state[
                                "retrieval"
                            ].context.text
                        ),
                        "previous_answer": (
                            state[
                                "generated"
                            ].model_dump_json(
                                indent=2
                            )
                        ),
                        "validation_errors": (
                            "\n".join(
                                f"- {error}"
                                for error in (
                                    validation.errors
                                )
                            )
                        ),
                    }
                )
            )

            if not isinstance(
                repaired,
                GeneratedAnswer,
            ):
                raise TypeError(
                    "修复结果不是 GeneratedAnswer："
                    f"{type(repaired)!r}"
                )

            logger.info(
                "[RAG] generation attempt=2 completed "
                "elapsed_ms=%.1f citations=%d",
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000.0,
                len(repaired.citations),
            )

        except Exception as error:
            logger.exception(
                "[RAG] generation attempt=2 failed"
            )
            errors.append(
                format_exception(
                    stage="引用修复生成",
                    error=error,
                )
            )

            return {
                "attempts": attempts,
                "validation_errors": errors,
                "answer": generation_failed_answer(),
            }

        return {
            "generated": repaired,
            "attempts": attempts,
            "validation_errors": errors,
        }

    def validate_repaired(
        state: AnswerGraphState,
    ) -> dict[str, object]:
        emit_progress(
            state.get("progress_callback"),
            "citation_validation",
            "正在校验修复后的证据编号…",
        )

        validation = (
            deps.citation_validator.validate(
                answer=state["generated"],
                context=state["retrieval"].context,
            )
        )

        logger.info(
            "[RAG] citation validation completed "
            "attempt=2 valid=%s citations=%d errors=%d",
            validation.is_valid,
            len(validation.citations),
            len(validation.errors),
        )

        if validation.is_valid:
            return {"validation": validation}

        logger.warning(
            "[RAG] citation validation failed "
            "attempt=2 errors=%s",
            validation.errors,
        )

        errors = list(
            state.get("validation_errors", [])
        )
        errors.extend(
            f"修复生成：{error}"
            for error in validation.errors
        )

        return {
            "validation": validation,
            "validation_errors": errors,
        }

    def finalize(
        state: AnswerGraphState,
    ) -> dict[str, object]:
        return {
            "answer": finalize_generated_answer(
                generated=state["generated"],
                validation=state["validation"],
            )
        }

    def insufficient(
        state: AnswerGraphState,
    ) -> dict[str, object]:
        return {"answer": insufficient_answer()}

    def generation_failed(
        state: AnswerGraphState,
    ) -> dict[str, object]:
        return {
            "answer": generation_failed_answer()
        }

    builder = StateGraph(AnswerGraphState)

    builder.add_node("prepare", prepare)
    builder.add_node("retrieve", retrieve)
    builder.add_node(
        "check_evidence", check_evidence
    )
    builder.add_node("generate", generate)
    builder.add_node(
        "validate_initial", validate_initial
    )
    builder.add_node("repair", repair)
    builder.add_node(
        "validate_repaired", validate_repaired
    )
    builder.add_node("finalize", finalize)
    builder.add_node(
        "insufficient", insufficient
    )
    builder.add_node(
        "generation_failed", generation_failed
    )

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "retrieve")
    builder.add_edge("retrieve", "check_evidence")

    builder.add_conditional_edges(
        "check_evidence",
        route_after_evidence,
        {
            "generate": "generate",
            "insufficient": "insufficient",
        },
    )
    builder.add_edge("insufficient", END)

    builder.add_conditional_edges(
        "generate",
        route_after_generate,
        {
            "validate_initial": "validate_initial",
            "generation_failed": "generation_failed",
        },
    )

    builder.add_conditional_edges(
        "validate_initial",
        route_after_initial_validation,
        {
            "finalize": "finalize",
            "repair": "repair",
            "generation_failed": "generation_failed",
        },
    )

    builder.add_conditional_edges(
        "repair",
        route_after_repair,
        {
            "validate_repaired": "validate_repaired",
            "generation_failed": "generation_failed",
        },
    )

    builder.add_conditional_edges(
        "validate_repaired",
        route_after_repaired_validation,
        {
            "finalize": "finalize",
            "generation_failed": "generation_failed",
        },
    )

    builder.add_edge("finalize", END)
    builder.add_edge("generation_failed", END)

    return builder.compile()
