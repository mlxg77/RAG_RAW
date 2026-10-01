"""阶段 4 运行日志与敏感信息脱敏。"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from rag_law.schemas import (
    AnswerPipelineResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ANSWER_LOG_PATH = (
    PROJECT_ROOT
    / "logs"
    / "answer_runs.jsonl"
)


SENSITIVE_PATTERNS = (
    (
        "email",
        re.compile(
            r"[A-Za-z0-9._%+-]+"
            r"@[A-Za-z0-9.-]+"
            r"\.[A-Za-z]{2,}"
        ),
        "[邮箱已脱敏]",
    ),
    (
        "id_card",
        re.compile(
            r"(?<!\d)"
            r"(?:\d{17}[\dXx]|\d{15})"
            r"(?!\d)"
        ),
        "[身份证号已脱敏]",
    ),
    (
        "mobile_phone",
        re.compile(
            r"(?<!\d)"
            r"(?:\+?86[- ]?)?"
            r"1[3-9]\d{9}"
            r"(?!\d)"
        ),
        "[手机号已脱敏]",
    ),
    (
        "bank_card",
        re.compile(
            r"(?<!\d)"
            r"(?:\d[ -]?){15,18}\d"
            r"(?!\d)"
        ),
        "[银行卡号已脱敏]",
    ),
    (
        "landline_phone",
        re.compile(
            r"(?<!\d)"
            r"0\d{2,3}[- ]?\d{7,8}"
            r"(?!\d)"
        ),
        "[电话号码已脱敏]",
    ),
)


def redact_text(
    text: str,
) -> tuple[str, set[str]]:
    """脱敏一段文本，并返回命中的敏感信息类型。"""

    redacted = text
    detected_types: set[str] = set()

    for (
        sensitive_type,
        pattern,
        replacement,
    ) in SENSITIVE_PATTERNS:
        redacted, replacement_count = (
            pattern.subn(
                replacement,
                redacted,
            )
        )

        if replacement_count:
            detected_types.add(
                sensitive_type
            )

    return redacted, detected_types


def redact_value(
    value: Any,
    *,
    detected_types: set[str],
) -> Any:
    """递归脱敏 JSON 兼容对象中的所有字符串。"""

    if isinstance(value, str):
        redacted, new_types = redact_text(
            value
        )
        detected_types.update(new_types)
        return redacted

    if isinstance(value, list):
        return [
            redact_value(
                item,
                detected_types=detected_types,
            )
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            redact_value(
                item,
                detected_types=detected_types,
            )
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: redact_value(
                item,
                detected_types=detected_types,
            )
            for key, item in value.items()
        }

    return value


class AnswerRunLogger:
    """将脱敏后的回答运行记录追加到 JSONL。"""

    def __init__(
        self,
        *,
        path: Path = DEFAULT_ANSWER_LOG_PATH,
        model_name: str,
    ) -> None:
        self.path = path
        self.model_name = model_name
        self._lock = Lock()

    @classmethod
    def from_settings(
        cls,
    ) -> "AnswerRunLogger":
        """使用项目模型配置创建默认日志器。"""

        from rag_law.config import settings

        return cls(
            model_name=settings.llm_model,
        )

    def build_record(
        self,
        *,
        result: AnswerPipelineResult,
        law_name: str | None = None,
    ) -> dict[str, Any]:
        """构造不含敏感信息和密钥的日志记录。"""

        detected_types: set[str] = set()

        redacted_query = redact_value(
            result.query,
            detected_types=detected_types,
        )
        redacted_law_name = redact_value(
            law_name,
            detected_types=detected_types,
        )

        retrieval_results = [
            {
                "rank": rank,
                "chunk_id": item.chunk_id,
                "law_id": item.law_id,
                "law_name": item.law_name,
                "article_no": item.article_no,
                "score": item.score,
                "source": item.source,
            }
            for rank, item in enumerate(
                result.retrieval.reranked_results,
                start=1,
            )
        ]

        retrieval_results = redact_value(
            retrieval_results,
            detected_types=detected_types,
        )

        answer_data = redact_value(
            result.answer.model_dump(
                mode="json"
            ),
            detected_types=detected_types,
        )

        return {
            "schema_version": "1.0",
            "run_id": str(uuid4()),
            "created_at": (
                datetime.now(timezone.utc)
                .isoformat()
            ),
            "query": redacted_query,
            "law_filter": redacted_law_name,
            "prompt_version": (
                result.prompt_version
            ),
            "model_name": self.model_name,
            "status": result.answer.status,
            "retrieval_results": (
                retrieval_results
            ),
            "answer": answer_data,
            "metrics": {
                "top_relevance_score": (
                    result.top_relevance_score
                ),
                "generation_attempts": (
                    result.generation_attempts
                ),
                "retrieval_latency_ms": (
                    result.retrieval
                    .retrieval_latency_ms
                ),
                "context_latency_ms": (
                    result.retrieval
                    .context_latency_ms
                ),
                "generation_latency_ms": (
                    result.generation_latency_ms
                ),
                "total_latency_ms": (
                    result.total_latency_ms
                ),
                "validation_error_count": len(
                    result.validation_errors
                ),
            },
            "redaction_types": sorted(
                detected_types
            ),
        }

    def write(
        self,
        *,
        result: AnswerPipelineResult,
        law_name: str | None = None,
    ) -> None:
        """将一条记录安全追加到 JSONL 文件。"""

        record = self.build_record(
            result=result,
            law_name=law_name,
        )

        serialized = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with self._lock:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with self.path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(serialized)
                file.write("\n")