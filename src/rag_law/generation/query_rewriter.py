"""多轮追问的检索查询改写。

把历史原文直接拼成检索输入会把本轮意图稀释：拼接模板词
（此前 / 用户 / 本轮）在 BM25 里是高 IDF 噪声，语义也会被前几轮
问题平均，只给条号的短追问最容易因此掉出候选池。这里改成先让
轻量模型把追问补成一句可独立检索的查询，改写不可用时再退回原来
的拼接方案，因此改写失败不会影响主链路。
"""

import logging
from typing import Protocol

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from rag_law.generation.chain import build_chat_model
from rag_law.generation.conversation import (
    PreparedConversation,
    build_contextualized_retrieval_query,
)
from rag_law.retrieval.article_anchor import (
    normalize_article_numbers,
)


logger = logging.getLogger(__name__)

# 改写策略版本；改写行为（提示词或归一化规则）变化时递增，
# 便于用历史日志对照「是改写变了还是召回变了」。
QUERY_REWRITE_VERSION = "query-rewrite-v1"

# 改写输入只收录用户问过的话，助手的长回答不参与。
MAX_REWRITE_INPUT_CHARS = 2000
# 正常输出是一句话；超过这个长度说明模型在夹带解释，按失败处理。
MAX_REWRITTEN_QUERY_CHARS = 120
DEFAULT_REWRITE_TIMEOUT_SECONDS = 12.0


REWRITE_SYSTEM_PROMPT = """
你负责把法律问答中的当前问题改写成一句可以独立检索的查询。

规则：
1. 只补全当前问题省略的指代和主语，例如省略的法律名称、条号。
2. 保留当前问题的原始意图，不要新增问题里没有的条件、事实或法律结论。
3. 不要解释、不要回答、不要扩写问题。
4. 条号一律写成「第N条」，N 用中文数字，例如「第二条」「第一百零五条」。
5. 当前问题本身已经完整时，原样输出。
6. 只输出改写后的一句话，不要加引号、标签、前缀或说明。
""".strip()


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_SYSTEM_PROMPT),
        (
            "human",
            """
<限定法律>{law_name}</限定法律>
<早期对话摘要>
{summary}
</早期对话摘要>
<此前用户问题>
{previous_questions}
</此前用户问题>
<当前问题>{question}</当前问题>

只输出改写后的查询本身。
            """.strip(),
        ),
    ]
)


_OUTPUT_PREFIXES = (
    "改写后的查询：",
    "改写后查询：",
    "检索查询：",
    "查询：",
)


class QueryRewriterProtocol(Protocol):
    """检索查询改写器；只依赖已经准备好的会话，不关心历史存在哪。"""

    def rewrite(
        self,
        question: str,
        conversation: PreparedConversation,
        *,
        law_name: str | None = None,
    ) -> str:
        ...


def normalize_rewritten_query(text: str) -> str:
    """清理模型输出：去掉前后缀与引号，统一条号形态。

    返回空字符串表示这条改写不可用，调用方应退回拼接方案。
    """

    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    cleaned = cleaned.splitlines()[0].strip()

    for prefix in _OUTPUT_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break

    cleaned = cleaned.strip("\"'“”‘’《》 　")
    cleaned = normalize_article_numbers(cleaned)

    if not cleaned:
        return ""

    # 超长说明模型在解释问题而不是改写问题，宁可退回旧方案。
    if len(cleaned) > MAX_REWRITTEN_QUERY_CHARS:
        return ""

    return cleaned


def _format_rewrite_input(
    question: str,
    conversation: PreparedConversation,
    *,
    law_name: str | None,
) -> dict[str, str]:
    """只把用户问过的话喂给改写模型，助手回答不参与。"""

    previous_questions = [
        turn.content
        for turn in conversation.recent_turns
        if turn.role == "user"
    ]
    previous_text = "\n".join(
        f"- {content}" for content in previous_questions
    )
    # 越早的问题越可能是噪音，超出长度时保留离本轮最近的。
    previous_text = previous_text[-MAX_REWRITE_INPUT_CHARS:]

    return {
        "law_name": law_name or "（未限定）",
        "summary": conversation.summary or "（无）",
        "previous_questions": previous_text or "（无）",
        "question": question.strip(),
    }


class LLMQueryRewriter:
    """用辅助任务的小模型改写检索查询。"""

    def __init__(
        self,
        chain: Runnable,
        *,
        version: str = QUERY_REWRITE_VERSION,
    ) -> None:
        self.chain = chain
        self.version = version

    @classmethod
    def load(cls) -> "LLMQueryRewriter":
        from rag_law.config import settings

        # 改写是小任务：用摘要模型即可，并单独收紧超时，
        # 避免一次慢调用把整轮问答拖住。
        model = build_chat_model(
            model=settings.resolved_summary_model(),
            timeout=(
                settings.query_rewrite_timeout_seconds
            ),
        )
        return cls(
            REWRITE_PROMPT
            | model
            | StrOutputParser()
        )

    def rewrite(
        self,
        question: str,
        conversation: PreparedConversation,
        *,
        law_name: str | None = None,
    ) -> str:
        result = self.chain.invoke(
            _format_rewrite_input(
                question,
                conversation,
                law_name=law_name,
            )
        )

        if not isinstance(result, str):
            raise ValueError(
                "查询改写模型返回了非文本结果"
            )

        return normalize_rewritten_query(result)


def build_retrieval_query(
    question: str,
    conversation: PreparedConversation,
    *,
    rewriter: QueryRewriterProtocol | None = None,
    law_name: str | None = None,
) -> str:
    """生成本轮真正送去检索的查询。

    首轮提问不需要改写，改写器缺失、报错或返回不可用内容时
    一律退回历史拼接，保证检索侧永远拿得到一个查询。
    """

    if rewriter is None or not conversation.recent_turns:
        return build_contextualized_retrieval_query(
            question,
            conversation,
        )

    try:
        rewritten = rewriter.rewrite(
            question,
            conversation,
            law_name=law_name,
        )
    except Exception as error:
        logger.warning(
            "[RAG] query rewrite failed type=%s; "
            "falling back to concatenation",
            type(error).__name__,
        )
        return build_contextualized_retrieval_query(
            question,
            conversation,
        )

    if not rewritten.strip():
        logger.warning(
            "[RAG] query rewrite returned empty; "
            "falling back to concatenation"
        )
        return build_contextualized_retrieval_query(
            question,
            conversation,
        )

    logger.info(
        "[RAG] query rewritten version=%s chars=%d",
        getattr(rewriter, "version", "unknown"),
        len(rewritten),
    )
    return rewritten
