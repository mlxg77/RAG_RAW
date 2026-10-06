from typing import Any

from langchain_core.runnables import (
    RunnableLambda,
)

from rag_law.generation.chain import (
    STRUCTURED_OUTPUT_METHOD,
    build_generation_chains,
)
from rag_law.schemas import (
    GeneratedAnswer,
    GeneratedCitation,
)


class FakeStructuredOutputModel:
    """不访问网络的结构化模型替身。"""

    def __init__(self) -> None:
        self.schema: type | None = None
        self.options: dict[str, Any] = {}
        self.prompt_values: list[Any] = []

    def with_structured_output(
        self,
        schema: type,
        **kwargs: Any,
    ) -> RunnableLambda:
        self.schema = schema
        self.options = kwargs

        return RunnableLambda(
            self._invoke
        )

    def _invoke(
        self,
        prompt_value: Any,
    ) -> GeneratedAnswer:
        self.prompt_values.append(
            prompt_value
        )

        return GeneratedAnswer(
            summary=(
                "建立劳动关系应当订立"
                "书面劳动合同。"
            ),
            analysis=(
                "检索证据说明了书面劳动合同要求。"
            ),
            conditions=[
                "需要确认双方是否已经建立劳动关系。"
            ],
            citations=[
                GeneratedCitation(
                    evidence_id="E001",
                )
            ],
            limitations=(
                "回答仅基于当前知识库，"
                "不能替代正式法律意见。"
            ),
            follow_up_question=None,
        )


def make_context() -> str:
    return (
        "[E001]\n"
        "法律：《中华人民共和国劳动合同法》\n"
        "条号：第十条\n"
        "证据类型：直接命中\n"
        "正文：\n"
        "建立劳动关系，应当订立书面劳动合同。"
    )


def test_build_chains_configures_structured_output() -> None:
    model = FakeStructuredOutputModel()

    chains = build_generation_chains(
        model
    )

    assert chains.answer_chain is not None
    assert chains.repair_chain is not None

    assert model.schema is GeneratedAnswer
    assert (
        model.options["method"]
        == STRUCTURED_OUTPUT_METHOD
    )
    assert (
        model.options["include_raw"]
        is False
    )


def test_answer_chain_returns_generated_answer() -> None:
    model = FakeStructuredOutputModel()
    chains = build_generation_chains(
        model
    )

    answer = chains.answer_chain.invoke(
        {
            "question": (
                "公司没有和我签劳动合同怎么办？"
            ),
            "context": make_context(),
        }
    )

    assert isinstance(
        answer,
        GeneratedAnswer,
    )
    assert answer.citations[0].evidence_id == "E001"

    assert len(model.prompt_values) == 1

    messages = (
        model.prompt_values[0].to_messages()
    )
    human_content = messages[1].content

    assert isinstance(human_content, str)
    assert "公司没有和我签劳动合同" in human_content
    assert "[E001]" in human_content


def test_repair_chain_receives_validation_errors() -> None:
    model = FakeStructuredOutputModel()
    chains = build_generation_chains(
        model
    )

    answer = chains.repair_chain.invoke(
        {
            "question": (
                "公司没有和我签劳动合同怎么办？"
            ),
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

    assert isinstance(
        answer,
        GeneratedAnswer,
    )

    assert len(model.prompt_values) == 1

    messages = (
        model.prompt_values[0].to_messages()
    )
    human_content = messages[1].content

    assert isinstance(human_content, str)
    assert "E999" in human_content
    assert "不存在的证据编号" in human_content
    assert "[E001]" in human_content
