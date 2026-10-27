"""阶段 4 的结构化答案生成链与流式生成通道。

回答模型换成了推理模型（如 deepseek-reasoner）。推理模型不支持强制
``tool_choice``，因此结构化输出统一改为 JSON 文本模式：提示词里写明
JSON 契约，模型输出文本，再由本地 Pydantic 校验兜底。

流式生成没有走 LangChain：它的流式抽象会丢掉 ``reasoning_content``，
而思考过程正是要展示给用户的内容，所以这里直接用 OpenAI 兼容客户端。
"""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import ValidationError

from rag_law.generation.partial_json import PartialFieldTracker
from rag_law.generation.prompts import (
    ANSWER_PROMPT,
    REPAIR_PROMPT,
)
from rag_law.schemas import GeneratedAnswer


STRUCTURED_OUTPUT_METHOD = "json_mode"
DEFAULT_LLM_TIMEOUT_SECONDS = 180.0
DEFAULT_LLM_MAX_RETRIES = 2
# 打字机节流：攒够这些字符或间隔到点，才向前端推一帧。
STREAM_FLUSH_CHARS = 8
STREAM_FLUSH_SECONDS = 0.05
# 思考过程更碎，单独用一套阈值，免得一秒刷上百帧。
REASONING_FLUSH_CHARS = 48
REASONING_FLUSH_SECONDS = 0.1
# 需要边生成边展示的字段；顺序即展示顺序。
STREAMED_FIELDS = ("summary", "analysis")
# 模型偶尔会给 JSON 套一层代码块围栏。
_CODE_FENCE = "```"
_ROLE_MAP = {"system": "system", "human": "user", "ai": "assistant"}


class JsonOutputModelProtocol(Protocol):
    """生成链所依赖的最小模型接口：能绑定 response_format 即可。"""

    def bind(self, **kwargs: Any) -> Runnable:
        ...


class EmptyGenerationError(RuntimeError):
    """模型返回了空内容；部分兼容服务偶发，允许原样重试一次。"""


@dataclass(frozen=True)
class GenerationStreamHandlers:
    """流式生成向外汇报的两类事件，两个回调都可缺省。"""

    # 思考过程的增量文本。
    on_reasoning: Callable[[str], None] | None = None
    # (字段名, 该字段当前的全量文本)；前端直接覆盖，不必自己拼接。
    on_field: Callable[[str, str], None] | None = None


@dataclass(frozen=True)
class GenerationStreamResult:
    answer: GeneratedAnswer
    reasoning: str


@dataclass(frozen=True)
class GenerationChains:
    """首次生成链、引用修复链，以及可选的流式生成通道。"""

    answer_chain: Runnable
    repair_chain: Runnable
    streamer: "AnswerStreamer | None" = None


def parse_generated_answer(text: str) -> GeneratedAnswer:
    """把模型返回的 JSON 文本解析成结构化回答。"""

    payload = _load_json_object(text)
    try:
        return GeneratedAnswer.model_validate(payload)
    except ValidationError as error:
        raise ValueError(
            f"结构化回答字段不符合要求：{error}"
        ) from error


def _load_json_object(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith(_CODE_FENCE):
        cleaned = cleaned[len(_CODE_FENCE) :]
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        if cleaned.endswith(_CODE_FENCE):
            cleaned = cleaned[: -len(_CODE_FENCE)]
        cleaned = cleaned.strip()

    if not cleaned:
        raise EmptyGenerationError("模型没有返回任何内容")

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"模型输出不是合法 JSON：{error.msg}（位置 {error.pos}）"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError("模型输出的 JSON 顶层必须是对象")
    return payload


def build_chat_model(
    *,
    model: str | None = None,
    timeout: float | None = None,
) -> ChatOpenAI:
    """根据项目配置创建模型客户端。

    把配置导入放在函数内部，使单元测试不必依赖真实 .env。
    辅助任务（如查询改写）可以传更短的 timeout，
    避免一个小调用把整轮问答拖到回答模型的等待上限。
    """

    from rag_law.config import settings

    return ChatOpenAI(
        model=model or settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0,
        timeout=timeout or settings.llm_timeout_seconds,
        max_retries=DEFAULT_LLM_MAX_RETRIES,
    )


def build_generation_chains(
    llm: JsonOutputModelProtocol,
) -> GenerationChains:
    """将 Prompt、JSON 模式模型和本地解析组成生成链。"""

    json_llm = llm.bind(response_format={"type": "json_object"})

    return GenerationChains(
        answer_chain=(
            ANSWER_PROMPT
            | json_llm
            | StrOutputParser()
            | RunnableLambda(parse_generated_answer)
        ),
        repair_chain=(
            REPAIR_PROMPT
            | json_llm
            | StrOutputParser()
            | RunnableLambda(parse_generated_answer)
        ),
        streamer=_build_streamer(llm),
    )


def load_generation_chains() -> GenerationChains:
    """加载真实模型并创建生成链。"""

    return build_generation_chains(
        build_chat_model()
    )


def _build_streamer(
    llm: JsonOutputModelProtocol,
) -> "AnswerStreamer | None":
    """从 LangChain 模型实例上取连接参数，构造流式通道。

    拿不到连接信息（例如测试替身）时返回 None，调用方自动退回非流式。
    """

    base_url = getattr(llm, "openai_api_base", None) or getattr(
        llm, "base_url", None
    )
    api_key = getattr(llm, "openai_api_key", None)
    model = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    if api_key is not None and hasattr(api_key, "get_secret_value"):
        api_key = api_key.get_secret_value()

    if not base_url or not api_key or not model:
        return None

    timeout = getattr(llm, "request_timeout", None) or (
        DEFAULT_LLM_TIMEOUT_SECONDS
    )

    return AnswerStreamer(
        client=OpenAI(
            api_key=str(api_key),
            base_url=str(base_url),
            timeout=timeout,
            max_retries=DEFAULT_LLM_MAX_RETRIES,
        ),
        prompt=ANSWER_PROMPT,
        model=str(model),
    )


class AnswerStreamer:
    """逐字生成结构化回答，同时把思考过程与正文分段推出去。"""

    def __init__(
        self,
        *,
        client: Any,
        prompt: ChatPromptTemplate,
        model: str,
    ) -> None:
        self.client = client
        self.prompt = prompt
        self.model = model

    def stream(
        self,
        payload: dict[str, Any],
        handlers: GenerationStreamHandlers | None = None,
    ) -> GenerationStreamResult:
        """流式生成一次回答；空结果允许静默重试一次。"""

        messages = _to_api_messages(
            self.prompt.format_messages(**payload)
        )

        try:
            return self._run(messages, handlers)
        except EmptyGenerationError:
            # 重试仍然推送正文，否则前端的打字机会停在半途；
            # 思考过程不再重复，因为上一轮的已经推过了。
            retry_handlers = (
                GenerationStreamHandlers(on_field=handlers.on_field)
                if handlers is not None
                else None
            )
            return self._run(messages, retry_handlers)

    def _run(
        self,
        messages: Sequence[dict[str, str]],
        handlers: GenerationStreamHandlers | None,
    ) -> GenerationStreamResult:
        reasoning_parts: list[str] = []
        pending_reasoning: list[str] = []
        reasoning_buffered = 0
        last_reasoning_flush = 0.0
        tracker = PartialFieldTracker(STREAMED_FIELDS)
        sent: dict[str, str] = {}
        buffered = 0
        last_flush = 0.0

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            response_format={"type": "json_object"},
            stream=True,
            temperature=0,
        )

        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta

            reasoning = getattr(delta, "reasoning_content", None) or ""
            if reasoning:
                reasoning_parts.append(reasoning)
                pending_reasoning.append(reasoning)
                reasoning_buffered += len(reasoning)
                now = time.monotonic()
                if (
                    reasoning_buffered >= REASONING_FLUSH_CHARS
                    or now - last_reasoning_flush >= REASONING_FLUSH_SECONDS
                ):
                    _emit_reasoning(pending_reasoning, handlers)
                    reasoning_buffered = 0
                    last_reasoning_flush = now

            content = getattr(delta, "content", None) or ""
            if not content:
                continue

            buffered += len(content)
            fields = tracker.feed(content)
            now = time.monotonic()
            if (
                buffered >= STREAM_FLUSH_CHARS
                or now - last_flush >= STREAM_FLUSH_SECONDS
            ):
                _emit_fields(fields, sent, handlers)
                buffered = 0
                last_flush = now

        _emit_reasoning(pending_reasoning, handlers)
        _emit_fields(tracker.feed(""), sent, handlers)

        answer = parse_generated_answer(tracker.buffer)
        return GenerationStreamResult(
            answer=answer,
            reasoning="".join(reasoning_parts).strip(),
        )


def _emit_reasoning(
    pending: list[str],
    handlers: GenerationStreamHandlers | None,
) -> None:
    """把攒下的思考片段合并成一帧推出去。"""

    if not pending:
        return
    text = "".join(pending)
    pending.clear()
    _notify(handlers.on_reasoning if handlers else None, text)


def _emit_fields(
    fields: dict[str, str],
    sent: dict[str, str],
    handlers: GenerationStreamHandlers | None,
) -> None:
    """只推送发生变化且已开始出字的字段。"""

    if handlers is None or handlers.on_field is None:
        sent.update(fields)
        return

    for field in STREAMED_FIELDS:
        value = fields.get(field)
        if value is None or value == sent.get(field):
            continue
        sent[field] = value
        _notify(handlers.on_field, field, value)


def _notify(
    callback: Callable[..., None] | None,
    *payload: str,
) -> None:
    """界面回调失败不能中断生成。"""

    if callback is None:
        return
    try:
        callback(*payload)
    except Exception:  # noqa: BLE001 - 展示层的问题与生成无关
        pass


def _to_api_messages(
    messages: Sequence[Any],
) -> list[dict[str, str]]:
    """把 LangChain 消息转换成 OpenAI 兼容接口的消息结构。"""

    converted: list[dict[str, str]] = []
    for message in messages:
        role = _ROLE_MAP.get(getattr(message, "type", ""), "user")
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = str(content)
        converted.append({"role": role, "content": content})
    return converted
