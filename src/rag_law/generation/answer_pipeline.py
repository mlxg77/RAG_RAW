"""阶段 4 完整回答流水线，编排由 LangGraph 驱动。

对外仍暴露 AnswerPipeline.run / load，检索、生成链、引用校验器
与运行日志保持原有注入方式，方便测试替换。
"""

import logging
import time
from collections.abc import Callable, Sequence

from rag_law.generation.answer_graph import (
    STANDARD_LIMITATION,
    build_answer_graph,
    clarification_answer,
    emit_progress,
    finalize_generated_answer,
    format_exception,
    generation_failed_answer,
    has_sufficient_evidence,
    insufficient_answer,
    normalize_conditions,
    normalize_follow_up,
    top_score,
)
from rag_law.generation.chain import (
    GenerationChains,
    load_generation_chains,
)
from rag_law.generation.citation_validator import (
    CitationValidator,
)
from rag_law.generation.conversation import (
    ConversationContextBuilder,
    ConversationSummaryCacheProtocol,
)
from rag_law.generation.prompts import (
    ANSWER_PROMPT_VERSION,
)
from rag_law.generation.run_logger import (
    AnswerRunLogger,
)
from rag_law.retrieval.retrieval_context_pipeline import (
    RetrievalContextPipeline,
)
from rag_law.schemas import (
    AnswerPipelineResult,
    AnswerResponse,
    CitationValidationResult,
    ConversationTurn,
    GeneratedAnswer,
    RetrievalContextResult,
)


logger = logging.getLogger(__name__)

# 接收 (str, str) 两个参数，返回 None。
# 回调只是“通知”，不需要回复。
ProgressCallback = Callable[[str, str], None]

# 临时阈值：
# 当前 24 道可回答题中，已知召回失败的 q019 Top1 约为 0.018；
# 其余题的最低 Top1 约为 0.238。
# 阶段 5 必须使用独立评测集重新校准。
DEFAULT_MIN_ANSWER_SCORE = 0.15


class AnswerPipeline:
    """检索、生成、引用校验和安全降级。"""

    def __init__(
        self,
        *,
        retrieval_pipeline,
        generation_chains: GenerationChains,
        citation_validator,
        conversation_context_builder: (
            ConversationContextBuilder | None
        ) = None,
        run_logger=None,
        min_answer_score: float = (
            DEFAULT_MIN_ANSWER_SCORE
        ),
    ) -> None:
        if not 0.0 <= min_answer_score <= 1.0:
            raise ValueError(
                "min_answer_score 必须在 [0, 1] 范围内"
            )

        self.retrieval_pipeline = retrieval_pipeline
        self.generation_chains = generation_chains
        self.citation_validator = citation_validator
        self.conversation_context_builder = (
            conversation_context_builder
            or ConversationContextBuilder(
                recent_rounds=3,
                summary_max_chars=1500,
                summarizer=None,
            )
        )
        self.run_logger = run_logger
        self.min_answer_score = min_answer_score

        # 图节点在运行时读取上面的属性，
        # 因此测试替换组件后无需重新编译。
        self._graph = build_answer_graph(self)

    @classmethod
    def load(
        cls,
        *,
        min_answer_score: float = (
            DEFAULT_MIN_ANSWER_SCORE
        ),
    ) -> "AnswerPipeline":
        """加载阶段 3 和阶段 4 的全部真实组件。"""

        return cls(
            retrieval_pipeline=(
                RetrievalContextPipeline.load()
            ),
            generation_chains=(
                load_generation_chains()
            ),
            citation_validator=(
                CitationValidator.load()
            ),
            conversation_context_builder=(
                ConversationContextBuilder.load()
            ),
            run_logger=(
                AnswerRunLogger.from_settings()
            ),
            min_answer_score=min_answer_score,
        )

    # ---------- 兼容旧接口的静态委托 ----------

    @staticmethod
    def _emit_progress(
        callback: ProgressCallback | None,
        stage: str,
        message: str,
    ) -> None:
        emit_progress(callback, stage, message)

    @staticmethod
    def _top_score(
        retrieval: RetrievalContextResult,
    ) -> float | None:
        return top_score(retrieval)

    def _has_sufficient_evidence(
        self,
        retrieval: RetrievalContextResult,
    ) -> bool:
        return has_sufficient_evidence(
            retrieval,
            self.min_answer_score,
        )

    @staticmethod
    def _normalize_follow_up(
        value: str | None,
    ) -> str | None:
        return normalize_follow_up(value)

    @staticmethod
    def _normalize_conditions(
        conditions: list[str],
    ) -> list[str]:
        return normalize_conditions(conditions)

    @staticmethod
    def _insufficient_answer() -> AnswerResponse:
        return insufficient_answer()

    @staticmethod
    def _generation_failed_answer() -> AnswerResponse:
        return generation_failed_answer()

    @staticmethod
    def _clarification_answer(
        generated: GeneratedAnswer,
    ) -> AnswerResponse:
        return clarification_answer(generated)

    @staticmethod
    def _finalize_generated_answer(
        *,
        generated: GeneratedAnswer,
        validation: CitationValidationResult,
    ) -> AnswerResponse:
        return finalize_generated_answer(
            generated=generated,
            validation=validation,
        )

    @staticmethod
    def _format_exception(
        *,
        stage: str,
        error: Exception,
    ) -> str:
        return format_exception(
            stage=stage,
            error=error,
        )

    @staticmethod
    def _build_result(
        *,
        query: str,
        answer: AnswerResponse,
        retrieval: RetrievalContextResult,
        generation_attempts: int,
        validation_errors: list[str],
        total_started_at: float,
        generation_started_at: float | None,
    ) -> AnswerPipelineResult:
        """构建最终结果：流水线所有出口共用的打包函数。"""

        if generation_started_at is None:
            generation_latency_ms = 0.0
        else:
            generation_latency_ms = (
                time.perf_counter()
                - generation_started_at
            ) * 1000.0

        total_latency_ms = (
            time.perf_counter()
            - total_started_at
        ) * 1000.0

        return AnswerPipelineResult(
            query=query,
            answer=answer,
            retrieval=retrieval,
            generation_attempts=(
                generation_attempts
            ),
            top_relevance_score=top_score(
                retrieval
            ),
            validation_errors=validation_errors,
            prompt_version=ANSWER_PROMPT_VERSION,
            generation_latency_ms=(
                generation_latency_ms
            ),
            total_latency_ms=total_latency_ms,
        )

    def run(
        self,
        query: str,
        *,
        law_name: str | None = None,
        history: Sequence[ConversationTurn] | None = None,
        summary_cache: ConversationSummaryCacheProtocol | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AnswerPipelineResult:
        """执行完整流水线，并记录脱敏运行日志。"""

        logger.info(
            "[RAG] request started query_chars=%d "
            "law_filter=%s history_turns=%d",
            len(query.strip()),
            law_name or "ALL",
            len(history or []),
        )

        try:
            result = self._run(
                query,
                law_name=law_name,
                history=history,
                summary_cache=summary_cache,
                progress_callback=progress_callback,
            )
        except Exception:
            self._emit_progress(
                progress_callback,
                "error",
                "处理失败，请查看控制台日志。",
            )
            logger.exception("[RAG] request failed")
            raise

        if self.run_logger is not None:
            try:
                self.run_logger.write(
                    result=result,
                    law_name=law_name,
                )
            except (
                OSError,
                TypeError,
                ValueError,
            ):
                # 日志失败不能让一个已经完成的法律回答丢失。
                # 异常只写入应用自身 stderr，不放进用户回答。
                logger.exception(
                    "写入回答运行日志失败"
                )

        logger.info(
            "[RAG] request completed status=%s attempts=%d "
            "retrieval_ms=%.1f context_ms=%.1f generation_ms=%.1f total_ms=%.1f",
            result.answer.status,
            result.generation_attempts,
            result.retrieval.retrieval_latency_ms,
            result.retrieval.context_latency_ms,
            result.generation_latency_ms,
            result.total_latency_ms,
        )
        self._emit_progress(
            progress_callback,
            "complete",
            "处理完成，正在展示结果。",
        )

        return result

    def _run(
        self,
        query: str,
        *,
        law_name: str | None = None,
        history: Sequence[ConversationTurn] | None = None,
        summary_cache: ConversationSummaryCacheProtocol | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AnswerPipelineResult:
        """执行阶段 4 完整流水线，不负责写运行日志。"""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("查询问题不能为空")

        total_started_at = time.perf_counter()

        state = self._graph.invoke(
            {
                "query": normalized_query,
                "law_name": law_name,
                "history": history,
                "summary_cache": summary_cache,
                "min_answer_score": (
                    self.min_answer_score
                ),
                "progress_callback": progress_callback,
                "total_started_at": total_started_at,
                "attempts": 0,
                "validation_errors": [],
                "generation_started_at": None,
            }
        )

        return self._build_result(
            query=normalized_query,
            answer=state["answer"],
            retrieval=state["retrieval"],
            generation_attempts=state.get(
                "attempts", 0
            ),
            validation_errors=state.get(
                "validation_errors", []
            ),
            total_started_at=total_started_at,
            generation_started_at=state.get(
                "generation_started_at"
            ),
        )
