"""登录、登出与当前用户接口，以及业务接口共用的登录门禁。"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from rag_law.auth.state import AuthState, AuthUnavailable, Session

router = APIRouter(prefix="/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=64)


class CurrentUser(BaseModel):
    id: int
    username: str
    role: str


def _auth_state(request: Request) -> AuthState:
    return request.app.state.auth


def require_user(request: Request) -> Session:
    """业务接口的登录门禁：没带 Cookie 或会话已失效一律 401。"""

    auth = _auth_state(request)
    sid = request.cookies.get(auth.cookie_name)
    if not sid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "请先登录")
    session = auth.sessions.get(sid)
    if session is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "登录已过期，请重新登录"
        )
    return session


@router.post("/login", response_model=CurrentUser)
def login(
    payload: LoginRequest, request: Request, response: Response
) -> CurrentUser:
    auth = _auth_state(request)

    try:
        user = auth.authenticate(payload.username, payload.password)
    except AuthUnavailable:
        # 具体原因（DSN 缺失、连不上库）只进服务端日志，不回给浏览器。
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "登录服务暂时不可用，请稍后再试",
        ) from None

    # 用户不存在与密码错误返回同一句话，避免账号枚举。
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "账号或密码错误")

    if user["status"] != 1:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "账号已被禁用")

    sid = auth.sessions.create(user["id"], user["username"], user["role"])
    response.set_cookie(
        key=auth.cookie_name,
        value=sid,
        max_age=auth.session_ttl_seconds,
        httponly=True,
        samesite="lax",
        secure=auth.cookie_secure,
        path="/",
    )
    return CurrentUser(id=user["id"], username=user["username"], role=user["role"])


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    auth = _auth_state(request)
    sid = request.cookies.get(auth.cookie_name)
    if sid:
        auth.sessions.delete(sid)
    response.delete_cookie(auth.cookie_name, path="/")
    return {"detail": "已退出登录"}


@router.get("/me", response_model=CurrentUser)
def me(session: Session = Depends(require_user)) -> CurrentUser:
    return CurrentUser(
        id=session.user_id, username=session.username, role=session.role
    )
