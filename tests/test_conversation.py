from rag_law.generation.conversation import (
    ConversationContextBuilder,
    build_contextualized_retrieval_query,
)
from rag_law.schemas import ConversationTurn


class StubSummarizer:
    def __init__(self, result: str = "用户此前讨论了劳动合同问题。") -> None:
        self.result = result
        self.calls: list[list[ConversationTurn]] = []

    def summarize(self, turns, *, max_chars):
        self.calls.append(list(turns))
        return self.result[:max_chars]


def _round(index: int) -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user", content=f"第 {index} 轮问题"),
        ConversationTurn(role="assistant", content=f"第 {index} 轮回答"),
    ]


def test_recent_rounds_are_configurable_and_older_turns_are_summarized() -> None:
    summarizer = StubSummarizer()
    builder = ConversationContextBuilder(
        recent_rounds=2,
        summary_max_chars=100,
        summarizer=summarizer,
    )
    history = [*_round(1), *_round(2), *_round(3), *_round(4)]

    prepared = builder.prepare(history)

    assert [turn.content for turn in prepared.recent_turns] == [
        "第 3 轮问题",
        "第 3 轮回答",
        "第 4 轮问题",
        "第 4 轮回答",
    ]
    assert [turn.content for turn in summarizer.calls[0]] == [
        "第 1 轮问题",
        "第 1 轮回答",
        "第 2 轮问题",
        "第 2 轮回答",
    ]
    assert "用户此前讨论了劳动合同问题" in prepared.prompt_text


def test_history_within_limit_does_not_call_summarizer() -> None:
    summarizer = StubSummarizer()
    builder = ConversationContextBuilder(
        recent_rounds=3,
        summary_max_chars=100,
        summarizer=summarizer,
    )

    prepared = builder.prepare([*_round(1), *_round(2)])

    assert prepared.summary == ""
    assert len(prepared.recent_turns) == 4
    assert summarizer.calls == []


def test_retrieval_query_combines_summary_recent_users_and_current_question() -> None:
    builder = ConversationContextBuilder(
        recent_rounds=1,
        summary_max_chars=100,
        summarizer=StubSummarizer("用户曾说明公司没有签劳动合同。"),
    )
    prepared = builder.prepare([*_round(1), *_round(2)])

    query = build_contextualized_retrieval_query("那赔偿怎么算？", prepared)

    assert "公司没有签劳动合同" in query
    assert "第 2 轮问题" in query
    assert "那赔偿怎么算" in query
    assert "第 2 轮回答" not in query


def test_summary_failure_falls_back_without_breaking_request() -> None:
    class BrokenSummarizer:
        def summarize(self, turns, *, max_chars):
            raise RuntimeError("模型暂不可用")

    builder = ConversationContextBuilder(
        recent_rounds=1,
        summary_max_chars=100,
        summarizer=BrokenSummarizer(),
    )

    prepared = builder.prepare([*_round(1), *_round(2)])

    assert "第 1 轮问题" in prepared.summary
    assert "第 2 轮问题" in prepared.prompt_text
