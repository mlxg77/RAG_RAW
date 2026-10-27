import json
from typing import Any

import pytest
from langchain_core.runnables import (
    RunnableLambda,
)

from rag_law.generation.chain import (
    STRUCTURED_OUTPUT_METHOD,
    build_generation_chains,
    parse_generated_answer,
)


def make_payload() -> str:
    return json.dumps(
        {
            "summary": "建立劳动关系应当订立书面劳动合同。",
            "analysis": "检索证据说明了书面劳动合同要求。",
            "conditions": ["需要确认双方是否已经建立劳动关系。"],
            "citations": [{"evidence_id": "E001"}],
            "limitations": "回答仅基于当前知识库，不能替代正式法律意见。",
            "follow_up_question": None,
        },
        ensure_ascii=False,
    )


class FakeJsonModel:
    """不访问网络的 JSON 模式模型替身。"""

    def __init__(self, output: str | None = None) -> None:
        self.options: dict[str, Any] = {}
        self.prompt_values: list[Any] = []
        self.output = output if output is not None else make_payload()

    def bind(self, **kwargs: Any) -> RunnableLambda:
        self.options.update(kwargs)

        return RunnableLambda(
            self._invoke
        )

    def _invoke(
        self,
        prompt_value: Any,
    ) -> str:
        self.prompt_values.append(
            prompt_value
        )

        return self.output


def make_context() -> str:
    return (
        "[E001]\n"
        "法律：《中华人民共和国劳动合同法》\n"
        "条号：第十条\n"
        "证据类型：直接命中\n"
        "正文：\n"
        "建立劳动关系，应当订立书面劳动合同。"
    )


def test_build_chains_uses_json_mode_output() -> None:
    model = FakeJsonModel()

    chains = build_generation_chains(
        model
    )

    assert chains.answer_chain is not None
    assert chains.repair_chain is not None

    assert STRUCTURED_OUTPUT_METHOD == "json_mode"
    assert model.options["response_format"] == {"type": "json_object"}

    # 替身没有连接信息，不应该构造出流式通道。
    assert chains.streamer is None


def test_answer_chain_returns_generated_answer() -> None:
    model = FakeJsonModel()
    chains = build_generation_chains(
        model
    )

    answer = chains.answer_chain.invoke(
        {
            "question": (
                "公司没有和我签劳动合同怎么办？"
            ),
            "conversation_context": "（没有最近对话，本轮为独立问题）",
            "context": make_context(),
        }
    )

    assert answer.citations[0].evidence_id == "E001"
    assert answer.follow_up_question is None

    assert len(model.prompt_values) == 1

    messages = (
        model.prompt_values[0].to_messages()
    )
    human_content = messages[1].content

    assert isinstance(human_content, str)
    assert "公司没有和我签劳动合同" in human_content
    assert "[E001]" in human_content

    system_content = messages[0].content
    assert isinstance(system_content, str)
    assert "一般法律规则" in system_content
    assert "不得追问其个人情况" in system_content
    assert "实时效力信息不在" in system_content
    assert "只输出一个 JSON 对象" in system_content


def test_repair_chain_receives_validation_errors() -> None:
    model = FakeJsonModel()
    chains = build_generation_chains(
        model
    )

    answer = chains.repair_chain.invoke(
        {
            "question": (
                "公司没有和我签劳动合同怎么办？"
            ),
            "conversation_context": "用户此前询问过劳动合同问题。",
            "context": make_context(),
            "previous_answer": (
                '{"citations": ['
                '{"evidence_id": "E999"}'
                "]}"
            ),
            "validation_errors": (
                "第 1 个引用使用了不存在的"
                "证据编号：E999"
            ),
        }
    )

    assert answer.citations[0].evidence_id == "E001"

    assert len(model.prompt_values) == 1

    messages = (
        model.prompt_values[0].to_messages()
    )
    human_content = messages[1].content

    assert isinstance(human_content, str)
    assert "E999" in human_content
    assert "不存在的证据编号" in human_content
    assert "[E001]" in human_content


def test_parse_generated_answer_strips_code_fence() -> None:
    fenced = f"```json\n{make_payload()}\n```"

    answer = parse_generated_answer(fenced)

    assert answer.summary.startswith("建立劳动关系")


def test_parse_generated_answer_rejects_broken_json() -> None:
    with pytest.raises(ValueError, match="不是合法 JSON"):
        parse_generated_answer('{"summary": "只有一半')
