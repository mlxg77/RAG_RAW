"""流式生成通道：思考过程、打字机增量与空结果重试。"""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from rag_law.generation.chain import (
    AnswerStreamer,
    EmptyGenerationError,
    GenerationStreamHandlers,
)
from rag_law.generation.prompts import ANSWER_PROMPT


def make_answer_json() -> str:
    return json.dumps(
        {
            "summary": "建立劳动关系应当订立书面劳动合同。",
            "analysis": "检索证据只说明了书面形式要求。",
            "conditions": ["需要确认双方是否已经建立劳动关系。"],
            "citations": [{"evidence_id": "E001"}],
            "limitations": "回答仅基于当前知识库。",
            "follow_up_question": None,
        },
        ensure_ascii=False,
    )


def _chunk(
    *,
    reasoning: str | None = None,
    content: str | None = None,
) -> Any:
    delta = SimpleNamespace(
        reasoning_content=reasoning,
        content=content,
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class FakeCompletions:
    def __init__(self, scripts: list[list[Any]]) -> None:
        self.scripts = list(scripts)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        if not self.scripts:
            raise AssertionError("没有剩余的流式脚本")
        return iter(self.scripts.pop(0))


class FakeClient:
    def __init__(self, scripts: list[list[Any]]) -> None:
        self.completions = FakeCompletions(scripts)
        self.chat = SimpleNamespace(completions=self.completions)


def make_payload() -> dict[str, str]:
    return {
        "question": "公司没有和我签劳动合同怎么办？",
        "conversation_context": "（没有最近对话，本轮为独立问题）",
        "context": "[E001] 建立劳动关系，应当订立书面劳动合同。",
    }


def split_into_chunks(text: str, size: int = 5) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


def make_streamer(scripts: list[list[Any]]) -> tuple[AnswerStreamer, FakeClient]:
    client = FakeClient(scripts)
    return (
        AnswerStreamer(
            client=client,
            prompt=ANSWER_PROMPT,
            model="deepseek-reasoner",
        ),
        client,
    )


def test_streams_reasoning_and_answer_fields() -> None:
    body = make_answer_json()
    script = [_chunk(reasoning="先看证据"), _chunk(reasoning="再作答")]
    script += [_chunk(content=piece) for piece in split_into_chunks(body)]

    streamer, client = make_streamer([script])
    reasoning: list[str] = []
    fields: dict[str, str] = {}

    result = streamer.stream(
        make_payload(),
        GenerationStreamHandlers(
            on_reasoning=reasoning.append,
            on_field=lambda field, text: fields.__setitem__(field, text),
        ),
    )

    assert "".join(reasoning) == "先看证据再作答"
    assert result.reasoning == "先看证据再作答"

    assert result.answer.summary == "建立劳动关系应当订立书面劳动合同。"
    assert result.answer.citations[0].evidence_id == "E001"

    # 打字机展示的是全量快照，最后一定等于定稿内容。
    assert fields["summary"] == result.answer.summary
    assert fields["analysis"] == result.answer.analysis

    request = client.completions.calls[0]
    assert request["model"] == "deepseek-reasoner"
    assert request["stream"] is True
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][0]["role"] == "system"


def test_empty_output_is_retried_without_duplicating_ui_events() -> None:
    body = make_answer_json()
    empty_script = [_chunk(reasoning="这次什么都没写")]
    good_script = [_chunk(content=piece) for piece in split_into_chunks(body)]

    streamer, client = make_streamer([empty_script, good_script])
    reasoning: list[str] = []
    fields: dict[str, str] = {}

    result = streamer.stream(
        make_payload(),
        GenerationStreamHandlers(
            on_reasoning=reasoning.append,
            on_field=lambda field, text: fields.__setitem__(field, text),
        ),
    )

    assert result.answer.summary.startswith("建立劳动关系")
    # 首次尝试的思考过程已经推给前端，重试不再重复推送。
    assert reasoning == ["这次什么都没写"]
    assert set(fields) == {"summary", "analysis"}
    assert len(client.completions.calls) == 2


def test_empty_output_twice_raises() -> None:
    streamer, client = make_streamer(
        [[_chunk(content="")], [_chunk(content="")]]
    )

    with pytest.raises(EmptyGenerationError):
        streamer.stream(make_payload())

    assert len(client.completions.calls) == 2
