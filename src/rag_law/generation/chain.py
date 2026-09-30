"""阶段 4 的结构化答案生成链。"""

from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from rag_law.generation.prompts import (
    ANSWER_PROMPT,
    REPAIR_PROMPT,
)
from rag_law.schemas import GeneratedAnswer


STRUCTURED_OUTPUT_METHOD = "function_calling"
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0
DEFAULT_LLM_MAX_RETRIES = 2


class StructuredOutputModelProtocol(Protocol):
    """生成链所依赖的最小模型接口。"""

    def with_structured_output(
        self,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> Runnable:
        ...


@dataclass(frozen=True)
class GenerationChains:
    """首次生成链与引用修复链。"""

    answer_chain: Runnable
    repair_chain: Runnable


def build_chat_model() -> ChatOpenAI:
    """根据项目配置创建回答模型客户端。

    把配置导入放在函数内部，使单元测试不必依赖真实 .env。
    """

    from rag_law.config import settings

    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0,
        timeout=DEFAULT_LLM_TIMEOUT_SECONDS,
        max_retries=DEFAULT_LLM_MAX_RETRIES,
    )


def build_generation_chains(
    llm: StructuredOutputModelProtocol,
) -> GenerationChains:
    """将 Prompt、LLM 和 Pydantic 输出解析组成 LCEL 链。"""

    structured_llm = llm.with_structured_output(
        GeneratedAnswer,
        method=STRUCTURED_OUTPUT_METHOD,
        include_raw=False,
    )

    return GenerationChains(
        answer_chain=(
            ANSWER_PROMPT
            | structured_llm
        ),
        repair_chain=(
            REPAIR_PROMPT
            | structured_llm
        ),
    )


def load_generation_chains() -> GenerationChains:
    """加载真实模型并创建两条生成链。"""

    return build_generation_chains(
        build_chat_model()
    )