-- RAG Law 历史对话表
-- 执行：psql -h <主机> -U <用户> -d <库名> -f scripts/init_conversations_db.sql
--
-- 与 users 表同库：conversations.user_id 指向 users.id，删账号时会话一并清理。
-- 服务启动后第一次读写也会执行同样的建表语句（ConversationStore 内幂等 DDL），
-- 因此该脚本主要给没有 CREATE 权限的受限账号使用。

CREATE TABLE IF NOT EXISTS conversations (
    id                  BIGSERIAL PRIMARY KEY,
    user_id             BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- 为空表示还没起标题，首次提问时用问题前 24 字补上。
    title               VARCHAR(80),
    law_name            VARCHAR(200),
    -- 早期对话的增量摘要，由回答链路在需要时补写。
    summary             TEXT,
    -- 摘要已覆盖到的最后一条消息 seq，用来只总结新增部分。
    summary_covered_seq INTEGER      NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 已建过 conversations 的库靠这两句补列（可重复执行）。
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS summary TEXT;
ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS summary_covered_seq INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
    ON conversations (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT      NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    -- 同一会话内自增：同一事务写入的两条消息只靠时间戳无法保证先后。
    seq             INTEGER     NOT NULL,
    role            VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant')),
    -- 纯文本，用于回填多轮上下文。
    content         TEXT        NOT NULL,
    -- 助手消息的完整 AnswerPipelineResult，用于还原引用卡片等界面细节。
    payload         JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_conversation_messages_seq UNIQUE (conversation_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation
    ON conversation_messages (conversation_id, seq);
