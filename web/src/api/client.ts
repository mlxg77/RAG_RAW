import { friendlyErrorMessage } from '@/lib/presentation'
import type {
  AnswerPipelineResult,
  HealthResponse,
  LawOption,
  QueryRequest,
} from '@/types'

export const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')

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
): Promise<T> {
  let response: Response

  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch {
    throw new ApiError('连不上问答服务，请确认后端已经启动。')
  }

  if (!response.ok) {
    throw await parseApiError(response)
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
