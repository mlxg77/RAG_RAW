-- RAG Law 登录账号表
-- 执行：psql -h <主机> -U <用户> -d <库名> -f scripts/init_auth_db.sql
--
-- 注意：password 为明文存储，仅适用于当前阶段的内网/演示环境。
-- 后续接入注册、改密或对外发布前，必须改为存哈希（argon2id / bcrypt）。

CREATE TABLE IF NOT EXISTS users (
    id         BIGSERIAL PRIMARY KEY,
    username   VARCHAR(32)  NOT NULL,
    password   VARCHAR(64)  NOT NULL,
    role       VARCHAR(16)  NOT NULL DEFAULT 'user',
    status     SMALLINT     NOT NULL DEFAULT 1,   -- 1 正常 0 禁用
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 大小写不敏感的唯一约束：Admin 与 admin 视为同一个账号
CREATE UNIQUE INDEX IF NOT EXISTS uq_users_username ON users (lower(username));

-- 手工添加账号示例（用户名按 lower() 去重，密码上限 64 字符）：
-- INSERT INTO users (username, password, role)
-- VALUES ('your_name', 'your_password', 'user');
