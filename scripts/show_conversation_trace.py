"""打印一个会话里每一轮的「实际检索输入 → 模型看到的证据 → 模型引用」。

这些内容本来就随每轮结果一起写进了 ``conversation_messages.payload``，
本脚本只是把已有数据换成一眼能看的对照表，不会重新跑检索或生成。
因此排查「为什么这一轮拒答」时不需要再复现一次请求。

用法（项目根目录执行）：

    uv run python scripts/show_conversation_trace.py --conversation-id 5
    uv run python scripts/show_conversation_trace.py --conversation-id 5 --user-id 1
"""

import argparse
import unicodedata

import psycopg
from psycopg.rows import dict_row

from rag_law.config import settings


LABEL_WIDTH = 16


def _display_width(text: str) -> int:
    """中文占两个字符宽度，ljust 按码点算会对不齐。"""

    return sum(
        2
        if unicodedata.east_asian_width(char) in {"W", "F"}
        else 1
        for char in text
    )


def _field(label: str, value: str) -> str:
    padding = " " * max(
        0,
        LABEL_WIDTH - _display_width(label),
    )
    return f"  {label}{padding}：{value}"


def _continuation(value: str) -> str:
    return f"  {' ' * LABEL_WIDTH}  {value}"


def _refs(items: list[str]) -> str:
    return " · ".join(items) if items else "（空）"


def _print_retrieval_query(
    retrieval_query: str | None,
    question: str,
) -> None:
    """追问轮的检索输入是多行拼接文本，逐行打印才看得清。"""

    text = (retrieval_query or "").strip()
    if not text:
        print(
            _field(
                "实际检索 query",
                "（未记录：旧版本日志没有这个字段）",
            )
        )
        return

    if text == question.strip():
        print(
            _field(
                "实际检索 query",
                "与本轮问题相同（本轮未做多轮改写）",
            )
        )
        return

    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]
    print(_field("实际检索 query", f"共 {len(lines)} 行"))
    for line in lines:
        print(_continuation(line))


def _explain_insufficient(
    payload: dict,
) -> str:
    """同一个 status 有两个来源，用「有没有跑过生成」区分。"""

    if float(
        payload.get("generation_latency_ms") or 0.0
    ) <= 0.0:
        return (
            "证据闸门拦截："
            "检索分数不够或上下文为空，未调用模型"
        )

    return (
        "模型判定证据不支持："
        "已调用模型，但一条引用都没给出"
    )


def _print_round(
    number: int,
    question: str,
    payload: dict | None,
) -> None:
    print()
    print(f"第 {number} 轮")
    print(_field("用户输入", question))

    if not payload:
        print(
            _field(
                "记录",
                "本轮没有落库结果，无法还原检索与证据",
            )
        )
        return

    retrieval = payload.get("retrieval") or {}
    answer = payload.get("answer") or {}
    context = retrieval.get("context") or {}

    _print_retrieval_query(
        retrieval.get("query"),
        question,
    )

    reranked = [
        item.get("article_no")
        for item in (
            retrieval.get("reranked_results") or []
        )
    ]
    evidences = [
        item.get("article_no")
        for item in (
            context.get("evidences") or []
        )
    ]
    citations = [
        f"{item.get('law_name')}{item.get('article_no')}"
        for item in (answer.get("citations") or [])
    ]

    print(
        _field(
            "精排命中",
            f"{_refs(reranked)}（{len(reranked)} 条）",
        )
    )
    print(
        _field(
            "模型看到",
            f"{_refs(evidences)}（{len(evidences)} 条）",
        )
    )
    print(_field("模型引用", _refs(citations)))

    score = payload.get("top_relevance_score")
    print(
        _field(
            "top 分数",
            "-" if score is None else f"{score:.4f}",
        )
    )

    status = answer.get("status") or "（未知）"
    print(_field("状态", status))

    if status == "insufficient_evidence":
        print(
            _field(
                "拒答来源",
                _explain_insufficient(payload),
            )
        )


def _load_messages(
    *,
    conversation_id: int,
    user_id: int | None,
) -> tuple[dict | None, list[dict]]:
    """读会话头与全部消息；不给 user_id 时跳过归属校验。"""

    condition = "WHERE id = %s"
    parameters: list[int] = [conversation_id]

    if user_id is not None:
        condition += " AND user_id = %s"
        parameters.append(user_id)

    with psycopg.connect(
        settings.database_url,
        row_factory=dict_row,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id, user_id, title, law_name
                FROM conversations
                {condition}
                """,
                parameters,
            )
            conversation = cursor.fetchone()

            if conversation is None:
                return None, []

            cursor.execute(
                """
                SELECT seq, role, content, payload
                FROM conversation_messages
                WHERE conversation_id = %s
                ORDER BY seq
                """,
                (conversation_id,),
            )
            return conversation, cursor.fetchall()


def _pair_rounds(
    messages: list[dict],
) -> list[tuple[str, dict | None]]:
    """把一问一答配成一轮；助手那条的 payload 就是本轮结果。"""

    rounds: list[tuple[str, dict | None]] = []

    for index, message in enumerate(messages):
        if message["role"] != "user":
            continue

        follower = (
            messages[index + 1]
            if index + 1 < len(messages)
            else None
        )
        payload = (
            follower["payload"]
            if follower is not None
            and follower["role"] == "assistant"
            else None
        )
        rounds.append((message["content"], payload))

    return rounds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "打印某会话每轮的实际检索输入、"
            "模型看到的证据条号与引用"
        )
    )
    parser.add_argument(
        "--conversation-id",
        type=int,
        required=True,
        help="会话 id（侧栏会话或 conversations.id）",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="会话所属用户；省略时跳过归属校验，仅限本机排障",
    )
    arguments = parser.parse_args()

    if not settings.database_url.strip():
        print("RAG_LAW_DATABASE_URL 未配置，无法读取会话。")
        raise SystemExit(1)

    try:
        conversation, messages = _load_messages(
            conversation_id=arguments.conversation_id,
            user_id=arguments.user_id,
        )
    except psycopg.Error as error:
        print(
            "读取会话失败："
            f"{type(error).__name__}: {error}"
        )
        raise SystemExit(1) from error

    if conversation is None:
        print(
            f"会话 {arguments.conversation_id} 不存在，"
            "或不属于指定用户。"
        )
        raise SystemExit(1)

    print(
        f"会话 {conversation['id']} · "
        f"{conversation['title'] or '（无标题）'}"
    )
    if conversation["law_name"]:
        print(_field("限定法律", conversation["law_name"]))
    if arguments.user_id is None:
        print(
            _field(
                "提示",
                "未指定 --user-id，本次跳过归属校验",
            )
        )

    rounds = _pair_rounds(messages)
    if not rounds:
        print()
        print("这个会话还没有任何提问记录。")
        return

    for number, (question, payload) in enumerate(
        rounds,
        start=1,
    ):
        _print_round(number, question, payload)


if __name__ == "__main__":
    main()
