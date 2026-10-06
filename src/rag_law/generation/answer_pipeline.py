"""阶段 4 完整回答流水线。"""

import logging
import time
from collections.abc import Callable
from typing import Protocol

from rag_law.generation.chain import (
    GenerationChains,
    load_generation_chains,
)
from rag_law.generation.citation_validator import (
    CitationValidator,
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
    GeneratedAnswer,
    RetrievalContextResult,
)


logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]

# 临时阈值：
# 当前 24 道可回答题中，已知召回失败的 q019 Top1 约为 0.018；
# 其余题的最低 Top1 约为 0.238。
# 阶段 5 必须使用独立评测集重新校准。
DEFAULT_MIN_ANSWER_SCORE = 0.15


STANDARD_LIMITATION = (
    "本回答仅依据当前知识库中的检索证据，"
    "不能替代律师针对具体事实提供的正式法律意见。"
)


class RetrievalPipelineProtocol(Protocol):
    """回答流水线依赖的检索接口。"""

    def run(
        self,
        query: str,
        *,
        law_name: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> RetrievalContextResult:
        ...


class CitationValidatorProtocol(Protocol):
    """回答流水线依赖的引用校验接口。"""

    def validate(
        self,
        *,
        answer: GeneratedAnswer,
        context,
    ) -> CitationValidationResult:
        ...


class AnswerRunLoggerProtocol(Protocol):
    """回答流水线依赖的运行日志接口。"""

    def write(
        self,
        *,
        result: AnswerPipelineResult,
        law_name: str | None = None,
    ) -> None:
        ...


class AnswerPipeline:
    """检索、生成、引用校验和安全降级。"""

    def __init__(
        self,
        *,
        retrieval_pipeline: RetrievalPipelineProtocol,
        generation_chains: GenerationChains,
        citation_validator: CitationValidatorProtocol,
        run_logger: AnswerRunLoggerProtocol | None = None,
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
        self.run_logger = run_logger
        self.min_answer_score = min_answer_score

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
            run_logger=(
                AnswerRunLogger.from_settings()
            ),
            min_answer_score=min_answer_score,
        )

    @staticmethod
    def _emit_progress(
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

    @staticmethod
    def _top_score(
        retrieval: RetrievalContextResult,
    ) -> float | None:
        if not retrieval.reranked_results:
            return None

        return retrieval.reranked_results[0].score

    def _has_sufficient_evidence(
        self,
        retrieval: RetrievalContextResult,
    ) -> bool:
        top_score = self._top_score(retrieval)

        return (
            top_score is not None
            and top_score >= self.min_answer_score
            and bool(retrieval.context.evidences)
            and bool(retrieval.context.text.strip())
        )

    @staticmethod
    def _normalize_follow_up(
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip()

        return normalized or None

    @staticmethod
    def _normalize_conditions(
        conditions: list[str],
    ) -> list[str]:
        return [
            condition.strip()
            for condition in conditions
            if condition.strip()
        ]

    @staticmethod
    def _insufficient_answer() -> AnswerResponse:
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

    @staticmethod
    def _generation_failed_answer() -> AnswerResponse:
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

    @staticmethod
    def _clarification_answer(
        generated: GeneratedAnswer,
    ) -> AnswerResponse:
        """没有有效引用但存在追问时，只返回安全追问。"""

        follow_up = (
            AnswerPipeline._normalize_follow_up(
                generated.follow_up_question
            )
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
            conditions=(
                AnswerPipeline._normalize_conditions(
                    generated.conditions
                )
            ),
            citations=[],
            limitations=STANDARD_LIMITATION,
            follow_up_question=follow_up,
        )

    @staticmethod
    def _finalize_generated_answer(
        *,
        generated: GeneratedAnswer,
        validation: CitationValidationResult,
    ) -> AnswerResponse:
        """把模型回答转换成最终可展示回答。"""

        follow_up = (
            AnswerPipeline._normalize_follow_up(
                generated.follow_up_question
            )
        )

        # 没有引用时，不能展示模型生成的实质性法律分析。
        if not validation.citations:
            if follow_up is not None:
                return (
                    AnswerPipeline._clarification_answer(
                        generated
                    )
                )

            return AnswerPipeline._insufficient_answer()

        status = (
            "needs_clarification"
            if follow_up is not None
            else "answered"
        )

        return AnswerResponse(
            status=status,
            summary=generated.summary.strip(),
            analysis=generated.analysis.strip(),
            conditions=(
                AnswerPipeline._normalize_conditions(
                    generated.conditions
                )
            ),
            citations=validation.citations,
            limitations=(
                generated.limitations.strip()
                or STANDARD_LIMITATION
            ),
            follow_up_question=follow_up,
        )

    @staticmethod
    def _format_exception(
        *,
        stage: str,
        error: Exception,
    ) -> str:
        """生成内部诊断信息，不直接作为用户回答展示。"""

        return (
            f"{stage}失败："
            f"{type(error).__name__}: {error}"
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
            top_relevance_score=(
                AnswerPipeline._top_score(
                    retrieval
                )
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
        progress_callback: ProgressCallback | None = None,
    ) -> AnswerPipelineResult:
        """执行完整流水线，并记录脱敏运行日志。"""

        logger.info(
            "[RAG] request started query_chars=%d law_filter=%s",
            len(query.strip()),
            law_name or "ALL",
        )

        try:
            result = self._run(
                query,
                law_name=law_name,
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
        progress_callback: ProgressCallback | None = None,
    ) -> AnswerPipelineResult:
        """执行阶段 4 完整流水线，不负责写运行日志。"""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("查询问题不能为空")

        total_started_at = time.perf_counter()

        retrieval = self.retrieval_pipeline.run(
            normalized_query,
            law_name=law_name,
            progress_callback=lambda stage, message: self._emit_progress(
                progress_callback,
                stage,
                message,
            ),
        )
        logger.info(
            "[RAG] retrieval completed results=%d retrieval_ms=%.1f context_ms=%.1f",
            len(retrieval.reranked_results),
            retrieval.retrieval_latency_ms,
            retrieval.context_latency_ms,
        )

        self._emit_progress(
            progress_callback,
            "evidence_check",
            "正在评估证据充分性…",
        )

        if not self._has_sufficient_evidence(
            retrieval
        ):
            return self._build_result(
                query=normalized_query,
                answer=self._insufficient_answer(),
                retrieval=retrieval,
                generation_attempts=0,
                validation_errors=[],
                total_started_at=total_started_at,
                generation_started_at=None,
            )

        generation_started_at = (
            time.perf_counter()
        )
        generation_attempts = 0
        validation_errors: list[str] = []

        try:
            generation_attempts += 1
            attempt_started_at = time.perf_counter()
            self._emit_progress(
                progress_callback,
                "generation",
                "正在依据检索证据生成回答…",
            )

            generated = (
                self.generation_chains
                .answer_chain
                .invoke(
                    {
                        "question": normalized_query,
                        "context": (
                            retrieval.context.text
                        ),
                    }
                )
            )

            if not isinstance(
                generated,
                GeneratedAnswer,
            ):
                raise TypeError(
                    "首次生成结果不是 GeneratedAnswer："
                    f"{type(generated)!r}"
                )

            logger.info(
                "[RAG] generation attempt=1 completed elapsed_ms=%.1f citations=%d",
                (time.perf_counter() - attempt_started_at) * 1000.0,
                len(generated.citations),
            )

        except Exception as error:
            logger.exception(
                "[RAG] generation attempt=1 failed"
            )
            validation_errors.append(
                self._format_exception(
                    stage="首次结构化生成",
                    error=error,
                )
            )

            return self._build_result(
                query=normalized_query,
                answer=(
                    self._generation_failed_answer()
                ),
                retrieval=retrieval,
                generation_attempts=(
                    generation_attempts
                ),
                validation_errors=validation_errors,
                total_started_at=total_started_at,
                generation_started_at=(
                    generation_started_at
                ),
            )

        self._emit_progress(
            progress_callback,
            "citation_validation",
            "正在校验引用与原始法条…",
        )
        first_validation = (
            self.citation_validator.validate(
                answer=generated,
                context=retrieval.context,
            )
        )
        logger.info(
            "[RAG] citation validation completed attempt=1 valid=%s citations=%d errors=%d",
            first_validation.is_valid,
            len(first_validation.citations),
            len(first_validation.errors),
        )

        if first_validation.is_valid:
            return self._build_result(
                query=normalized_query,
                answer=(
                    self._finalize_generated_answer(
                        generated=generated,
                        validation=first_validation,
                    )
                ),
                retrieval=retrieval,
                generation_attempts=(
                    generation_attempts
                ),
                validation_errors=[],
                total_started_at=total_started_at,
                generation_started_at=(
                    generation_started_at
                ),
            )

        validation_errors.extend(
            f"首次生成：{error}"
            for error in first_validation.errors
        )
        logger.warning(
            "[RAG] citation validation failed attempt=1 errors=%s",
            first_validation.errors,
        )

        # 引用校验失败时只允许修复一次。
        try:
            generation_attempts += 1
            repair_started_at = time.perf_counter()
            self._emit_progress(
                progress_callback,
                "repair",
                "证据编号未通过校验，正在进行一次安全修复…",
            )

            repaired = (
                self.generation_chains
                .repair_chain
                .invoke(
                    {
                        "question": normalized_query,
                        "context": (
                            retrieval.context.text
                        ),
                        "previous_answer": (
                            generated.model_dump_json(
                                indent=2
                            )
                        ),
                        "validation_errors": (
                            "\n".join(
                                f"- {error}"
                                for error
                                in first_validation.errors
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
                "[RAG] generation attempt=2 completed elapsed_ms=%.1f citations=%d",
                (time.perf_counter() - repair_started_at) * 1000.0,
                len(repaired.citations),
            )

        except Exception as error:
            logger.exception(
                "[RAG] generation attempt=2 failed"
            )
            validation_errors.append(
                self._format_exception(
                    stage="引用修复生成",
                    error=error,
                )
            )

            return self._build_result(
                query=normalized_query,
                answer=(
                    self._generation_failed_answer()
                ),
                retrieval=retrieval,
                generation_attempts=(
                    generation_attempts
                ),
                validation_errors=validation_errors,
                total_started_at=total_started_at,
                generation_started_at=(
                    generation_started_at
                ),
            )

        self._emit_progress(
            progress_callback,
            "citation_validation",
            "正在校验修复后的证据编号…",
        )
        repaired_validation = (
            self.citation_validator.validate(
                answer=repaired,
                context=retrieval.context,
            )
        )
        logger.info(
            "[RAG] citation validation completed attempt=2 valid=%s citations=%d errors=%d",
            repaired_validation.is_valid,
            len(repaired_validation.citations),
            len(repaired_validation.errors),
        )

        if repaired_validation.is_valid:
            return self._build_result(
                query=normalized_query,
                answer=(
                    self._finalize_generated_answer(
                        generated=repaired,
                        validation=(
                            repaired_validation
                        ),
                    )
                ),
                retrieval=retrieval,
                generation_attempts=(
                    generation_attempts
                ),
                validation_errors=(
                    validation_errors
                ),
                total_started_at=total_started_at,
                generation_started_at=(
                    generation_started_at
                ),
            )

        validation_errors.extend(
            f"修复生成：{error}"
            for error
            in repaired_validation.errors
        )
        logger.warning(
            "[RAG] citation validation failed attempt=2 errors=%s",
            repaired_validation.errors,
        )

        return self._build_result(
            query=normalized_query,
            answer=(
                self._generation_failed_answer()
            ),
            retrieval=retrieval,
            generation_attempts=(
                generation_attempts
            ),
            validation_errors=validation_errors,
            total_started_at=total_started_at,
            generation_started_at=(
                generation_started_at
            ),
        )
