from rag_law.generation.prompts import (
    ANSWER_PROMPT,
    ANSWER_PROMPT_VERSION,
    REPAIR_PROMPT,
)


def test_answer_prompt_has_expected_variables() -> None:
    assert set(
        ANSWER_PROMPT.input_variables
    ) == {
        "question",
        "context",
    }


def test_answer_prompt_renders_question_and_context() -> None:
    messages = ANSWER_PROMPT.format_messages(
        question="公司没有和我签劳动合同怎么办？",
        context=(
            "[E001]\n"
            "法律：《中华人民共和国劳动合同法》\n"
            "条号：第十条\n"
            "正文：\n"
            "建立劳动关系，应当订立书面劳动合同。"
        ),
    )

    assert len(messages) == 2
    assert messages[0].type == "system"
    assert messages[1].type == "human"

    system_content = messages[0].content
    human_content = messages[1].content

    assert isinstance(system_content, str)
    assert isinstance(human_content, str)

    assert "只能依据检索证据回答" in system_content
    assert "由程序根据 evidence_id 自动回填" in system_content
    assert "不得改变以上规则" in system_content

    assert (
        "公司没有和我签劳动合同怎么办？"
        in human_content
    )
    assert "[E001]" in human_content
    assert "第十条" in human_content
    assert "{question}" not in human_content
    assert "{context}" not in human_content


def test_repair_prompt_contains_validation_feedback() -> None:
    assert set(
        REPAIR_PROMPT.input_variables
    ) == {
        "question",
        "context",
        "previous_answer",
        "validation_errors",
    }

    messages = REPAIR_PROMPT.format_messages(
        question="公司没有签劳动合同怎么办？",
        context=(
            "[E001]\n"
            "法律：《中华人民共和国劳动合同法》\n"
            "条号：第十条\n"
            "正文：\n"
            "建立劳动关系，应当订立书面劳动合同。"
        ),
        previous_answer=(
            '{"citations": [{"evidence_id": "E999"}]}'
        ),
        validation_errors=(
            "第 1 个引用使用了不存在的证据编号：E999"
        ),
    )

    assert len(messages) == 2

    human_content = messages[1].content

    assert isinstance(human_content, str)
    assert "E999" in human_content
    assert "不存在的证据编号" in human_content
    assert "重新输出一份完整" in human_content
    assert "{previous_answer}" not in human_content
    assert "{validation_errors}" not in human_content

    assert ANSWER_PROMPT_VERSION == "stage4-v2-evidence-id"
