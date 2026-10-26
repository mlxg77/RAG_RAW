"""历史对话的 PostgreSQL 存取。

与 ``AuthState`` 一样复用 ``RAG_LAW_DATABASE_URL``，连接池延迟到第一次
使用时才建立。区别在于这里的写入是「尽力而为」：落库失败只应该丢掉一条
历史，绝不能让已经生成好的回答变成 500。
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolClosed, PoolTimeout

from rag_law.schemas import AnswerPipelineResult

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "新对话"
MAX_TITLE_CHARS = 24
# 与 ConversationTurn.content 的上限保持一致：回填多轮上下文时直接可用。
MAX_CONTENT_CHARS = 2000

# 幂等建表：受限账号没有 CREATE 权限时会失败，此时改用 scripts/init_conversations_db.sql。
SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id                  BIGSERIAL PRIMARY KEY,
        user_id             BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title               VARCHAR(80),
        law_name            VARCHAR(200),
        -- 早期对话的增量摘要，由回答链路在需要时补写。
        summary             TEXT,
        -- 摘要已覆盖到的最后一条消息 seq。
        summary_covered_seq INTEGER      NOT NULL DEFAULT 0,
        created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
        updated_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
    )
    """,
    # 表已存在时上面的 CREATE 不会补列，这两条负责升级旧库。
    "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS summary TEXT",
    (
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS "
        "summary_covered_seq INTEGER NOT NULL DEFAULT 0"
    ),
    """
    CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
        ON conversations (user_id, updated_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_messages (
        id              BIGSERIAL PRIMARY KEY,
        conversation_id BIGINT      NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        seq             INTEGER     NOT NULL,
        role            VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant')),
        content         TEXT        NOT NULL,
        payload         JSONB,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_conversation_messages_seq UNIQUE (conversation_id, seq)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation
        ON conversation_messages (conversation_id, seq)
    """,
)


class ConversationStoreUnavailable(RuntimeError):
    """会话库未配置或连不上，调用方应转成 503。"""


class ConversationStore:
    """``conversations`` / ``conversation_messages`` 两张表的读写入口。"""

    def __init__(self, *, dsn: str, pool_max_size: int = 4) -> None:
        self._dsn = dsn.strip()
        self._pool_max_size = pool_max_size
        self._pool: ConnectionPool | None = None
        self._schema_ready = False
        self._lock = Lock()

    @property
    def configured(self) -> bool:
        return bool(self._dsn)

    # ---------- 写入 ----------

    def create(
        self,
        *,
        user_id: int,
        law_name: str | None = None,
    ) -> dict[str, Any]:
        """建一个空会话；标题留空，等第一次提问再补。"""

        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (user_id, law_name)
                    VALUES (%s, %s)
                    RETURNING id, title, law_name, created_at, updated_at
                    """,
                    (user_id, law_name),
                )
                row = cur.fetchone()
        return _summary(row)

    def append_exchange(
        self,
        *,
        conversation_id: int,
        user_id: int,
        query: str,
        law_name: str | None,
        result: AnswerPipelineResult,
    ) -> bool:
        """把一次成功的问答写成两条消息；会话不属于本人时返回 False。"""

        # UPDATE 同时校验归属并锁住会话行，同一会话的并发写入会自然排队，
        # 从而让下面基于 MAX(seq) 的编号不会撞车。
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    UPDATE conversations
                    SET updated_at = now(),
                        title = COALESCE(title, %s),
                        law_name = COALESCE(law_name, %s)
                    WHERE id = %s AND user_id = %s
                    RETURNING id
                    """,
                    (_derive_title(query), law_name, conversation_id, user_id),
                )
                if cur.fetchone() is None:
                    return False

                cur.execute(
                    """
                    SELECT COALESCE(MAX(seq), 0) AS seq
                    FROM conversation_messages
                    WHERE conversation_id = %s
                    """,
                    (conversation_id,),
                )
                base_seq = cur.fetchone()["seq"]

                cur.execute(
                    """
                    INSERT INTO conversation_messages
                        (conversation_id, seq, role, content, payload)
                    VALUES (%s, %s, 'user', %s, NULL),
                           (%s, %s, 'assistant', %s, %s)
                    """,
                    (
                        conversation_id,
                        base_seq + 1,
                        query,
                        conversation_id,
                        base_seq + 2,
                        _answer_content(result),
                        Jsonb(result.model_dump(mode="json")),
                    ),
                )
        return True

    def delete(self, *, conversation_id: int, user_id: int) -> bool:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM conversations
                    WHERE id = %s AND user_id = %s
                    RETURNING id
                    """,
                    (conversation_id, user_id),
                )
                return cur.fetchone() is not None

    def save_summary(
        self,
        *,
        conversation_id: int,
        user_id: int,
        summary: str,
        covered_seq: int,
    ) -> bool:
        """写回早期对话摘要；不动 updated_at，避免摘要缓存把列表顺序刷乱。"""

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE conversations
                    SET summary = %s,
                        summary_covered_seq = %s
                    WHERE id = %s AND user_id = %s
                    """,
                    (summary, covered_seq, conversation_id, user_id),
                )
                return cur.rowcount > 0

    # ---------- 读取 ----------

    def load_messages(
        self,
        *,
        conversation_id: int,
        user_id: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """按 seq 取最近若干条消息，用于回填多轮上下文；不属于本人时返回空。"""

        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT 1 FROM conversations WHERE id = %s AND user_id = %s",
                    (conversation_id, user_id),
                )
                if cur.fetchone() is None:
                    return []

                cur.execute(
                    """
                    SELECT seq, role, content
                    FROM conversation_messages
                    WHERE conversation_id = %s
                    ORDER BY seq DESC
                    LIMIT %s
                    """,
                    (conversation_id, limit),
                )
                rows = cur.fetchall()
        return list(reversed(rows))

    def load_summary(
        self,
        *,
        conversation_id: int,
        user_id: int,
    ) -> tuple[str, int]:
        """返回 (已缓存的摘要, 摘要覆盖到的最后一条消息 seq)。"""

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT summary, summary_covered_seq
                    FROM conversations
                    WHERE id = %s AND user_id = %s
                    """,
                    (conversation_id, user_id),
                )
                row = cur.fetchone()

        if row is None:
            return "", 0
        return row[0] or "", int(row[1] or 0)

    def list_for_user(
        self,
        *,
        user_id: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """列出有内容的会话；点了「新对话」却没提问的空壳不占用列表。"""

        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT c.id, c.title, c.law_name,
                           c.created_at, c.updated_at,
                           count(m.id) AS message_count
                    FROM conversations c
                    LEFT JOIN conversation_messages m
                        ON m.conversation_id = c.id
                    WHERE c.user_id = %s
                    GROUP BY c.id
                    HAVING count(m.id) > 0
                    ORDER BY c.updated_at DESC, c.id DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
        return [_summary(row) for row in rows]

    def get(
        self,
        *,
        conversation_id: int,
        user_id: int,
    ) -> dict[str, Any] | None:
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT id, title, law_name, created_at, updated_at
                    FROM conversations
                    WHERE id = %s AND user_id = %s
                    """,
                    (conversation_id, user_id),
                )
                summary = cur.fetchone()
                if summary is None:
                    return None

                cur.execute(
                    """
                    SELECT id, role, content, payload, created_at
                    FROM conversation_messages
                    WHERE conversation_id = %s
                    ORDER BY seq
                    """,
                    (conversation_id,),
                )
                messages = cur.fetchall()

        detail = _summary(summary)
        detail["message_count"] = len(messages)
        detail["messages"] = [_message(row) for row in messages]
        return detail

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()

    # ---------- 连接 ----------

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        """借一条连接并在正常退出时提交；驱动与连接池异常统一转 503。"""

        try:
            pool = self._ensure_pool()
            with pool.connection() as conn:
                self._ensure_schema(conn)
                yield conn
        except (psycopg.Error, PoolTimeout, PoolClosed) as error:
            logger.warning("会话库不可用：%s", type(error).__name__)
            raise ConversationStoreUnavailable(type(error).__name__) from error

    def _ensure_pool(self) -> ConnectionPool:
        if not self.configured:
            raise ConversationStoreUnavailable("RAG_LAW_DATABASE_URL 未配置")

        with self._lock:
            if self._pool is None:
                self._pool = ConnectionPool(
                    conninfo=self._dsn,
                    min_size=1,
                    max_size=self._pool_max_size,
                    timeout=5.0,
                )
            return self._pool

    def _ensure_schema(self, conn: psycopg.Connection) -> None:
        """第一次使用时建表；已建过就只剩一次布尔判断。"""

        if self._schema_ready:
            return

        with self._lock:
            if self._schema_ready:
                return
            with conn.cursor() as cur:
                for statement in SCHEMA_STATEMENTS:
                    cur.execute(statement)
            self._schema_ready = True


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"] or DEFAULT_TITLE,
        "law_name": row["law_name"],
        "message_count": int(row.get("message_count") or 0),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _message(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "result": row["payload"],
        "created_at": row["created_at"],
    }


def _derive_title(query: str) -> str:
    compact = " ".join(query.split())
    return compact[:MAX_TITLE_CHARS] or DEFAULT_TITLE


def _answer_content(result: AnswerPipelineResult) -> str:
    """拼出与前端 buildConversationHistory 一致的多轮上下文文本。"""

    answer = result.answer
    parts = [answer.summary, answer.analysis]
    if answer.conditions:
        parts.append(f"适用条件：{'；'.join(answer.conditions)}")
    if answer.follow_up_question:
        parts.append(f"追问：{answer.follow_up_question}")

    text = "\n".join(part.strip() for part in parts if part and part.strip())
    if not text:
        return "（本次没有生成可复用的结论）"
    return text[:MAX_CONTENT_CHARS]
