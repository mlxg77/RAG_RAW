"""登录相关接口的行为测试；账号库用内存实现，不依赖真实 PostgreSQL。"""

from fastapi.testclient import TestClient

from rag_law.api.main import create_app
from rag_law.auth.state import AuthState


class StubAuthState(AuthState):
    def __init__(self) -> None:
        super().__init__(
            dsn="",
            cookie_name="rag_law_sid",
            session_ttl_seconds=600,
            cookie_secure=False,
        )
        self.users = {
            "admin": {
                "id": 1,
                "username": "admin",
                "password": "123456",
                "role": "admin",
                "status": 1,
            },
            "frozen": {
                "id": 2,
                "username": "frozen",
                "password": "123456",
                "role": "user",
                "status": 0,
            },
        }

    def authenticate(self, username: str, password: str) -> dict | None:
        user = self.users.get(username.lower())
        if user is None or user["password"] != password:
            return None
        return user


def _client() -> TestClient:
    return TestClient(create_app(auth=StubAuthState()))


def test_login_returns_user_and_sets_hardened_cookie() -> None:
    client = _client()

    response = client.post(
        "/v1/auth/login", json={"username": "admin", "password": "123456"}
    )

    assert response.status_code == 200
    assert response.json() == {"id": 1, "username": "admin", "role": "admin"}

    raw_cookie = response.headers["set-cookie"].lower()
    assert "rag_law_sid=" in raw_cookie
    assert "httponly" in raw_cookie
    assert "samesite=lax" in raw_cookie


def test_login_does_not_leak_whether_account_exists() -> None:
    client = _client()

    wrong_password = client.post(
        "/v1/auth/login", json={"username": "admin", "password": "nope"}
    )
    unknown_user = client.post(
        "/v1/auth/login", json={"username": "ghost", "password": "nope"}
    )

    assert wrong_password.status_code == 401
    assert unknown_user.status_code == 401
    assert wrong_password.json() == unknown_user.json()


def test_login_rejects_disabled_account() -> None:
    client = _client()

    response = client.post(
        "/v1/auth/login", json={"username": "frozen", "password": "123456"}
    )

    assert response.status_code == 403


def test_protected_endpoints_reject_anonymous_requests() -> None:
    client = _client()

    assert client.get("/v1/auth/me").status_code == 401
    assert client.get("/v1/laws").status_code == 401


def test_me_returns_current_user_after_login() -> None:
    client = _client()
    client.post(
        "/v1/auth/login", json={"username": "admin", "password": "123456"}
    )

    response = client.get("/v1/auth/me")

    assert response.status_code == 200
    assert response.json()["username"] == "admin"


def test_logout_invalidates_existing_session() -> None:
    client = _client()
    client.post(
        "/v1/auth/login", json={"username": "admin", "password": "123456"}
    )

    assert client.post("/v1/auth/logout").status_code == 200
    assert client.get("/v1/auth/me").status_code == 401
