"""历史对话模块：把会话与消息落到 PostgreSQL。"""

from rag_law.history.store import (
    ConversationStore,
    ConversationStoreUnavailable,
)

__all__ = [
    "ConversationStore",
    "ConversationStoreUnavailable",
]
