"""按 .env 里的 RAG_LAW_DATABASE_URL 初始化登录账号表。

用法：uv run python scripts/init_auth_db.py
执行内容见 scripts/init_auth_db.sql，可重复运行。
"""

from pathlib import Path

import psycopg

from rag_law.config import settings

SQL_PATH = Path(__file__).resolve().parent / "init_auth_db.sql"


def main() -> int:
    if not settings.database_url.strip():
        print("请先在 .env 里填写 RAG_LAW_DATABASE_URL，再重新运行本脚本。")
        return 1

    sql = SQL_PATH.read_text(encoding="utf-8")
    try:
        with psycopg.connect(settings.database_url) as conn:
            conn.execute(sql)
            conn.commit()

        with psycopg.connect(settings.database_url) as conn:
            rows = conn.execute(
                "SELECT username, role, status FROM users ORDER BY id"
            ).fetchall()
    except psycopg.Error as error:
        print(f"连库或建表失败：{error}")
        return 2

    print(f"账号表就绪，当前共 {len(rows)} 个账号：")
    for username, role, status in rows:
        print(f"  - {username}（{role}，status={status}）")
    if not rows:
        print("  （还没有账号，按 init_auth_db.sql 末尾的示例插入一条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
