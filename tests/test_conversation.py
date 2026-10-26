from rag_law.generation.conversation import (
    ConversationContextBuilder,
    build_contextualized_retrieval_query,
)
from rag_law.schemas import ConversationTurn


class StubSummarizer:
    def __init__(self, result: str = "用户此前讨论了劳动合同问题。") -> None:
        self.result = result
        self.calls: list[list[ConversationTurn]] = []

    def summarize(self, turns, *, max_chars, prior_summary=""):
        self.calls.append(list(turns))
        return self.result[:max_chars]


class MergingSummarizer(StubSummarizer):
    """记录传入的旧摘要，用来验证增量合并。"""

    def __init__(self, result: str = "更新后的摘要") -> None:
        super().__init__(result)
        self.priors: list[str] = []

    def summarize(self, turns, *, max_chars, prior_summary=""):
        self.priors.append(prior_summary)
        return super().summarize(turns, max_chars=max_chars)


class FailingSummarizer:
    def summarize(self, turns, *, max_chars, prior_summary=""):
        raise RuntimeError("模型不可用")


class StubSummaryCache:
    def __init__(self, summary: str = "", covered_seq: int = 0) -> None:
        self.summary = summary
        self.covered_seq = covered_seq
        self.saved: list[tuple[str, int]] = []

    def load(self) -> tuple[str, int]:
        return self.summary, self.covered_seq

    def save(self, summary: str, covered_seq: int) -> None:
        self.saved.append((summary, covered_seq))
        self.summary = summary
        self.covered_seq = covered_seq


def _round(index: int) -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user", content=f"第 {index} 轮问题"),
        ConversationTurn(role="assistant", content=f"第 {index} 轮回答"),
    ]


def _numbered_round(index: int, start_seq: int) -> list[ConversationTurn]:
    """带 seq 的一轮，模拟从历史对话表回填的结果。"""

    return [
        ConversationTurn(
            role="user", content=f"第 {index} 轮问题", seq=start_seq
        ),
        ConversationTurn(
            role="assistant",
            content=f"第 {index} 轮回答",
            seq=start_seq + 1,
        ),
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


def test_cached_summary_is_reused_without_calling_model() -> None:
    summarizer = StubSummarizer()
    builder = ConversationContextBuilder(
        recent_rounds=1,
        summary_max_chars=100,
        summarizer=summarizer,
    )
    cache = StubSummaryCache("旧摘要", 2)
    history = [*_numbered_round(1, 1), *_numbered_round(2, 3)]

    prepared = builder.prepare(history, summary_cache=cache)

    assert prepared.summary == "旧摘要"
    assert summarizer.calls == []
    assert cache.saved == []


def test_only_new_older_turns_are_summarized_and_cursor_advances() -> None:
    summarizer = MergingSummarizer("更新后的摘要")
    builder = ConversationContextBuilder(
        recent_rounds=1,
        summary_max_chars=100,
        summarizer=summarizer,
    )
    cache = StubSummaryCache("旧摘要", 2)
    history = [
        *_numbered_round(1, 1),
        *_numbered_round(2, 3),
        *_numbered_round(3, 5),
    ]

    prepared = builder.prepare(history, summary_cache=cache)

    assert prepared.summary == "更新后的摘要"
    assert [turn.content for turn in summarizer.calls[0]] == [
        "第 2 轮问题",
        "第 2 轮回答",
    ]
    assert summarizer.priors == ["旧摘要"]
    assert cache.saved == [("更新后的摘要", 4)]


def test_uncached_turns_are_summarized_even_without_seq() -> None:
    summarizer = StubSummarizer("新摘要")
    builder = ConversationContextBuilder(
        recent_rounds=1,
        summary_max_chars=100,
        summarizer=summarizer,
    )
    cache = StubSummaryCache("旧摘要", 99)

    prepared = builder.prepare([*_round(1), *_round(2)], summary_cache=cache)

    # 前端传来的历史没有 seq，无法证明已被覆盖，只能重新总结。
    assert prepared.summary == "新摘要"
    assert [turn.content for turn in summarizer.calls[0]] == [
        "第 1 轮问题",
        "第 1 轮回答",
    ]
    assert cache.saved == []


def test_summary_failure_does_not_advance_cache_cursor() -> None:
    builder = ConversationContextBuilder(
        recent_rounds=1,
        summary_max_chars=100,
        summarizer=FailingSummarizer(),
    )
    cache = StubSummaryCache("旧摘要", 0)
    history = [*_numbered_round(1, 1), *_numbered_round(2, 3)]

    prepared = builder.prepare(history, summary_cache=cache)

    assert "第 1 轮问题" in prepared.summary
    assert cache.saved == []
