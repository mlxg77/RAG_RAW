from typing import Any

import pytest

from rag_law.generation.conversation import (
    PreparedConversation,
)
from rag_law.generation.query_rewriter import (
    LLMQueryRewriter,
    MAX_REWRITTEN_QUERY_CHARS,
    build_retrieval_query,
    normalize_article_numbers,
    normalize_rewritten_query,
)
from rag_law.retrieval.article_anchor import (
    to_chinese_article_number,
)
from rag_law.schemas import ConversationTurn


def make_conversation() -> PreparedConversation:
    """复现「刑法第二十条 → 第一条呢 → 第二条呢」的第三轮。"""

    return PreparedConversation(
        summary="",
        recent_turns=(
            ConversationTurn(
                role="user",
                content="刑法第二十条是什么",
            ),
            ConversationTurn(
                role="assistant",
                content="第二十条规定了正当防卫。",
            ),
            ConversationTurn(
                role="user",
                content="第一条呢",
            ),
            ConversationTurn(
                role="assistant",
                content="第一条规定了立法目的。",
            ),
        ),
    )


class StubRewriter:
    version = "stub-v1"

    def __init__(
        self,
        result: str = "中华人民共和国刑法 第二条",
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    def rewrite(
        self,
        question: str,
        conversation: PreparedConversation,
        *,
        law_name: str | None = None,
    ) -> str:
        self.calls.append((question, law_name))
        if self.error is not None:
            raise self.error
        return self.result


class StubChain:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.payloads: list[dict[str, str]] = []

    def invoke(
        self,
        payload: dict[str, str],
    ) -> Any:
        self.payloads.append(payload)
        return self.result


def test_first_turn_skips_rewrite() -> None:
    rewriter = StubRewriter()

    query = build_retrieval_query(
        "劳动合同没签怎么赔偿？",
        PreparedConversation(
            summary="",
            recent_turns=(),
        ),
        rewriter=rewriter,
    )

    # 首轮没有指代可消解，多调一次模型纯属浪费延迟。
    assert query == "劳动合同没签怎么赔偿？"
    assert rewriter.calls == []


def test_follow_up_uses_rewritten_query() -> None:
    rewriter = StubRewriter()

    query = build_retrieval_query(
        "第二条呢",
        make_conversation(),
        rewriter=rewriter,
        law_name="中华人民共和国刑法",
    )

    assert query == "中华人民共和国刑法 第二条"
    # 法律限定要传下去，改写器才知道省略的是哪部法。
    assert rewriter.calls == [
        ("第二条呢", "中华人民共和国刑法")
    ]


def test_missing_rewriter_keeps_concatenation() -> None:
    query = build_retrieval_query(
        "第二条呢",
        make_conversation(),
    )

    assert query.startswith(
        "此前用户问题：刑法第二十条是什么"
    )
    assert query.endswith("本轮用户问题：第二条呢")


@pytest.mark.parametrize(
    "result",
    ["", "   ", "\n"],
)
def test_empty_rewrite_falls_back(
    result: str,
) -> None:
    rewriter = StubRewriter(result)

    query = build_retrieval_query(
        "第二条呢",
        make_conversation(),
        rewriter=rewriter,
    )

    assert query.endswith("本轮用户问题：第二条呢")


def test_rewriter_error_falls_back() -> None:
    rewriter = StubRewriter(
        error=TimeoutError("改写超时")
    )

    query = build_retrieval_query(
        "第二条呢",
        make_conversation(),
        rewriter=rewriter,
    )

    # 改写失败不能把整轮问答拖下水。
    assert query.endswith("本轮用户问题：第二条呢")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "一"),
        (2, "二"),
        (10, "十"),
        (11, "十一"),
        (20, "二十"),
        (101, "一百零一"),
        (105, "一百零五"),
        (110, "一百一十"),
        (120, "一百二十"),
        (1001, "一千零一"),
        (1260, "一千二百六十"),
    ],
)
def test_chinese_article_number(
    value: int,
    expected: str,
) -> None:
    assert to_chinese_article_number(value) == expected


def test_article_numbers_match_index_form() -> None:
    # 索引里的 article_no 是中文数字，阿拉伯数字必须归一。
    assert (
        normalize_article_numbers(
            "刑法第2条、第 20 条、第１０５条"
        )
        == "刑法第二条、第二十条、第一百零五条"
    )
    # 已经是中文形态时保持不变。
    assert (
        normalize_article_numbers("第一百零五条")
        == "第一百零五条"
    )


def test_normalize_strips_wrapping_noise() -> None:
    assert (
        normalize_rewritten_query(
            "  改写后的查询：“刑法第2条”  "
        )
        == "刑法第二条"
    )


def test_normalize_keeps_first_line_only() -> None:
    assert (
        normalize_rewritten_query(
            "中华人民共和国刑法 第二条\n"
            "这一条规定了刑法的任务。"
        )
        == "中华人民共和国刑法 第二条"
    )


def test_normalize_rejects_overlong_output() -> None:
    text = "中华人民共和国刑法第二条" * 20

    assert len(text) > MAX_REWRITTEN_QUERY_CHARS
    # 超长说明模型在解释而不是改写，宁可退回旧方案。
    assert normalize_rewritten_query(text) == ""


def test_rewrite_input_excludes_assistant_answers() -> None:
    chain = StubChain("中华人民共和国刑法 第二条")
    rewriter = LLMQueryRewriter(chain)

    rewritten = rewriter.rewrite(
        "第二条呢",
        make_conversation(),
    )

    assert rewritten == "中华人民共和国刑法 第二条"
    payload = chain.payloads[0]
    assert "刑法第二十条是什么" in payload[
        "previous_questions"
    ]
    # 助手回答又长又可能被当成事实依据，不能进改写输入。
    assert "第二十条规定了" not in payload[
        "previous_questions"
    ]
    assert payload["law_name"] == "（未限定）"


def test_llm_rewriter_rejects_non_text_output() -> None:
    rewriter = LLMQueryRewriter(StubChain(None))

    with pytest.raises(ValueError):
        rewriter.rewrite("第二条呢", make_conversation())
