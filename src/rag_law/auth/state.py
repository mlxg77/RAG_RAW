"""登录态容器：账号库连接池 + 进程内会话表。

连接池延迟到第一次登录请求才建，避免导入模块时就去连库——与
``ApiServices`` 的惰性加载保持一致的思路。会话只存在进程内，因此
uvicorn 重启后需要重新登录；部署脚本用的是 ``--workers 1``，单进程够用。
将来要多实例部署，把 ``SessionStore`` 换成 Redis 实现即可，路由层不用动。
"""

import logging
import secrets
import time
from dataclasses import dataclass
from threading import Lock

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolClosed, PoolTimeout

logger = logging.getLogger(__name__)


class AuthUnavailable(RuntimeError):
    """账号库未配置或连不上，调用方应转成 503。"""


@dataclass(frozen=True)
class Session:
    user_id: int
    username: str
    role: str
    expires_at: float


class SessionStore:
    """session_id -> 用户信息。Cookie 里只有随机串，不携带任何用户数据。"""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._data: dict[str, Session] = {}
        self._lock = Lock()

    def create(self, user_id: int, username: str, role: str) -> str:
        sid = secrets.token_urlsafe(32)
        with self._lock:
            self._purge_expired()
            self._data[sid] = Session(
                user_id=user_id,
                username=username,
                role=role,
                expires_at=time.time() + self._ttl,
            )
        return sid

    def get(self, sid: str) -> Session | None:
        with self._lock:
            session = self._data.get(sid)
            if session is None:
                return None
            if session.expires_at <= time.time():
                self._data.pop(sid, None)
                return None
            return session

    def delete(self, sid: str) -> None:
        with self._lock:
            self._data.pop(sid, None)

    def _purge_expired(self) -> None:
        now = time.time()
        stale = [sid for sid, item in self._data.items() if item.expires_at <= now]
        for sid in stale:
            self._data.pop(sid, None)


class AuthState:
    """挂在 ``app.state.auth`` 上的登录基础设施。"""

    def __init__(
        self,
        *,
        dsn: str,
        cookie_name: str,
        session_ttl_seconds: int,
        cookie_secure: bool,
    ) -> None:
        self._dsn = dsn.strip()
        self.cookie_name = cookie_name
        self.session_ttl_seconds = session_ttl_seconds
        self.cookie_secure = cookie_secure
        self.sessions = SessionStore(session_ttl_seconds)
        self._pool: ConnectionPool | None = None
        self._lock = Lock()

    @property
    def configured(self) -> bool:
        return bool(self._dsn)

    def authenticate(self, username: str, password: str) -> dict | None:
        """按用户名取用户并与库中密码比对；连不上库时抛 AuthUnavailable。"""

        if not self.configured:
            raise AuthUnavailable("RAG_LAW_DATABASE_URL 未配置")

        try:
            with self._ensure_pool().connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT id, username, password, role, status
                        FROM users
                        WHERE lower(username) = lower(%s)
                        """,
                        (username,),
                    )
                    row = cur.fetchone()
        except (psycopg.Error, PoolTimeout, PoolClosed) as error:
            # 连不上库属于部署问题，要能被日志看见，但不能泄露给浏览器。
            logger.warning("账号库不可用：%s", type(error).__name__)
            raise AuthUnavailable(type(error).__name__) from error

        if row is None or row["password"] != password:
            return None
        return row

    def _ensure_pool(self) -> ConnectionPool:
        with self._lock:
            if self._pool is None:
                self._pool = ConnectionPool(
                    conninfo=self._dsn,
                    min_size=1,
                    max_size=5,
                    timeout=5.0,
                )
            return self._pool

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()
