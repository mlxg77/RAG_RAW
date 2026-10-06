"""前端展示所需的纯函数，避免把展示规则散落在 Streamlit 页面中。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rag_law.schemas import AnswerPipelineResult, AnswerStatus, ValidatedCitation


@dataclass(frozen=True)
class LawOption:
    """法律筛选器中的一项。"""

    law_id: str
    law_name: str


@dataclass(frozen=True)
class StatusPresentation:
    """回答状态对应的用户可见文案。"""

    label: str
    tone: str
    description: str


STATUS_PRESENTATIONS: dict[AnswerStatus, StatusPresentation] = {
    "answered": StatusPresentation(
        label="已找到依据",
        tone="success",
        description="回答中的引用已经过程序校验。",
    ),
    "needs_clarification": StatusPresentation(
        label="需要补充信息",
        tone="warning",
        description="补充一个关键事实后，回答会更准确。",
    ),
    "insufficient_evidence": StatusPresentation(
        label="依据不足",
        tone="neutral",
        description="当前知识库没有提供足够证据。",
    ),
    "generation_failed": StatusPresentation(
        label="已安全停止",
        tone="danger",
        description="本次生成或引用校验未通过，未展示未经验证的内容。",
    ),
}


EXAMPLE_QUESTIONS = (
    "公司一直没和我签书面劳动合同，我可以要求什么？",
    "酒后驾驶机动车会受到什么处罚？",
    "消费者买到不符合食品安全标准的食品，可以要求赔偿吗？",
)


def load_law_options(manifest_path: Path) -> list[LawOption]:
    """从 manifest 读取稳定、有序的法律筛选项。"""

    with manifest_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    if not isinstance(payload, list):
        raise ValueError("法律清单必须是列表")

    options: list[LawOption] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("法律清单中的每一项必须是对象")

        law_id = str(item.get("law_id", "")).strip()
        law_name = str(item.get("law_name", "")).strip()
        if not law_id or not law_name:
            raise ValueError("法律清单缺少 law_id 或 law_name")

        options.append(LawOption(law_id=law_id, law_name=law_name))

    return options


def status_presentation(status: AnswerStatus) -> StatusPresentation:
    """返回状态的统一展示规则。"""

    return STATUS_PRESENTATIONS[status]


def citation_location(citation: ValidatedCitation) -> str:
    """把来源文件与页码组合成人类可读的位置。"""

    location = citation.source_file
    if citation.page is None:
        return location

    if citation.end_page is not None and citation.end_page != citation.page:
        return f"{location} · 第 {citation.page}–{citation.end_page} 页"

    return f"{location} · 第 {citation.page} 页"


def result_metrics(result: AnswerPipelineResult) -> dict[str, str]:
    """生成页面顶部的紧凑运行指标。"""

    return {
        "引用": str(len(result.answer.citations)),
        "总耗时": f"{result.total_latency_ms / 1000:.1f} 秒",
        "生成次数": str(result.generation_attempts),
    }


def to_session_payload(result: AnswerPipelineResult) -> dict[str, Any]:
    """把 Pydantic 结果转换成 Streamlit 会话可安全保存的普通对象。"""

    return result.model_dump(mode="json")


def from_session_payload(payload: dict[str, Any]) -> AnswerPipelineResult:
    """从会话数据恢复完整结果并重新执行类型校验。"""

    return AnswerPipelineResult.model_validate(payload)

