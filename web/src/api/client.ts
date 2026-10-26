import { friendlyErrorMessage } from '@/lib/presentation'
import type {
  AnswerPipelineResult,
  ConversationDetail,
  ConversationSummary,
  CurrentUser,
  HealthResponse,
  LawOption,
  LoginRequest,
  QueryRequest,
} from '@/types'

export const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')

let unauthorizedHandler: (() => void) | null = null

/** 业务请求碰到 401 时触发，用来把界面切回登录页。 */
export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler
}

export function triggerUnauthorized(): void {
  unauthorizedHandler?.()
}

interface RequestOptions {
  /** 登录与登录态探测本身用 401 表达业务结果，不应触发全局登出。 */
  skipUnauthorized?: boolean
}

export class ApiError extends Error {
  readonly status: number
  readonly detail: string

  constructor(
    message: string,
    options: { status?: number; detail?: string } = {},
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = options.status ?? 0
    this.detail = options.detail ?? ''
  }
}

export async function parseApiError(response: Response): Promise<ApiError> {
  let detail = ''

  try {
    const payload = (await response.json()) as { detail?: unknown }
    if (typeof payload.detail === 'string') {
      detail = payload.detail
    } else if (payload.detail !== undefined) {
      detail = JSON.stringify(payload.detail)
    }
  } catch {
    // 网关或代理返回的非 JSON 错误页没有可读细节。
  }

  return new ApiError(friendlyErrorMessage(response.status, detail), {
    status: response.status,
    detail,
  })
}

async function requestJson<T>(
  path: string,
  init?: RequestInit,
  options: RequestOptions = {},
): Promise<T> {
  let response: Response

  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch {
    throw new ApiError('连不上问答服务，请确认后端已经启动。')
  }

  if (!response.ok) {
    const error = await parseApiError(response)
    if (error.status === 401 && !options.skipUnauthorized) {
      triggerUnauthorized()
    }
    throw error
  }

  return (await response.json()) as T
}

export function fetchLaws(): Promise<LawOption[]> {
  return requestJson<LawOption[]>('/v1/laws')
}

export function fetchHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>('/v1/health')
}

export function queryAnswer(
  payload: QueryRequest,
): Promise<AnswerPipelineResult> {
  return requestJson<AnswerPipelineResult>('/v1/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function fetchConversations(): Promise<ConversationSummary[]> {
  return requestJson<ConversationSummary[]>('/v1/conversations')
}

export function createConversation(
  lawName: string | null,
): Promise<ConversationSummary> {
  return requestJson<ConversationSummary>('/v1/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ law_name: lawName }),
  })
}

export function fetchConversation(id: number): Promise<ConversationDetail> {
  return requestJson<ConversationDetail>(`/v1/conversations/${id}`)
}

export function deleteConversation(id: number): Promise<void> {
  return requestJson<{ detail: string }>(`/v1/conversations/${id}`, {
    method: 'DELETE',
  }).then(() => undefined)
}

export function login(payload: LoginRequest): Promise<CurrentUser> {
  return requestJson<CurrentUser>(
    '/v1/auth/login',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
    { skipUnauthorized: true },
  )
}

export function logout(): Promise<void> {
  return requestJson<{ detail: string }>('/v1/auth/logout', {
    method: 'POST',
  }).then(() => undefined)
}

export function fetchCurrentUser(): Promise<CurrentUser> {
  return requestJson<CurrentUser>('/v1/auth/me', undefined, {
    skipUnauthorized: true,
  })
}
