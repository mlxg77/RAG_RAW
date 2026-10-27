import { API_BASE, parseApiError, triggerUnauthorized } from '@/api/client'
import type {
  AnswerPipelineResult,
  AnswerTextField,
  ProgressEvent,
  QueryRequest,
  RebuildResult,
  StreamErrorPayload,
  UploadResult,
} from '@/types'

export interface KnowledgeStreamHandlers<T> {
  onProgress: (event: ProgressEvent) => void
  onComplete: (result: T) => void
  onError: (payload: StreamErrorPayload) => void
}

interface EventStreamHandlers<T> {
  onProgress: (event: ProgressEvent) => void
  /** progress 与终帧之外的中间事件，例如思考过程和正文增量。 */
  onEvent?: (event: string, data: unknown) => void
  onComplete: (result: T) => void
  onError: (payload: StreamErrorPayload) => void
}

interface StreamFrame {
  event: string
  data: unknown
}

function parseFrame(block: string): StreamFrame | null {
  let event = 'message'
  const dataLines: string[] = []

  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trim())
    }
  }

  if (dataLines.length === 0) {
    return null
  }

  try {
    return { event, data: JSON.parse(dataLines.join('\n')) }
  } catch {
    return null
  }
}

async function streamEventRequest<T>(
  path: string,
  init: RequestInit,
  handlers: EventStreamHandlers<T>,
): Promise<void> {
  let response: Response

  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch {
    handlers.onError({ message: '连不上服务，请确认后端已经启动。' })
    return
  }

  if (!response.ok) {
    const error = await parseApiError(response)
    if (response.status === 401) {
      triggerUnauthorized()
    }
    handlers.onError({ message: error.message, detail: error.detail })
    return
  }

  if (!response.body) {
    handlers.onError({
      message: '当前浏览器读不到处理进度，请换用 Chrome 或 Edge。',
    })
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finished = false

  const dispatch = (block: string) => {
    const frame = parseFrame(block)
    if (!frame) {
      return
    }

    if (frame.event === 'progress') {
      handlers.onProgress(frame.data as ProgressEvent)
    } else if (frame.event === 'complete') {
      finished = true
      handlers.onComplete(frame.data as T)
    } else if (frame.event === 'error') {
      finished = true
      handlers.onError(frame.data as StreamErrorPayload)
    } else {
      handlers.onEvent?.(frame.event, frame.data)
    }
  }

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) {
        break
      }

      buffer += decoder.decode(value, { stream: true })
      let boundary = buffer.indexOf('\n\n')
      while (boundary >= 0) {
        dispatch(buffer.slice(0, boundary))
        buffer = buffer.slice(boundary + 2)
        boundary = buffer.indexOf('\n\n')
      }
    }

    buffer += decoder.decode()
    if (buffer.trim()) {
      dispatch(buffer)
    }
  } catch {
    if (!finished) {
      handlers.onError({ message: '处理进度中断了，请刷新页面后重试。' })
      finished = true
    }
  } finally {
    reader.releaseLock()
  }

  if (!finished) {
    handlers.onError({ message: '服务没有返回最终结果，请稍后重试。' })
  }
}

function streamKnowledgeRequest<T>(
  path: string,
  body: FormData,
  handlers: KnowledgeStreamHandlers<T>,
): Promise<void> {
  return streamEventRequest<T>(
    path,
    {
      method: 'POST',
      body,
      headers: { Accept: 'text/event-stream' },
    },
    handlers,
  )
}

export interface AnswerStreamHandlers {
  onProgress: (event: ProgressEvent) => void
  /** 思考过程的增量文本，需要自己拼接。 */
  onReasoning: (text: string) => void
  /** 正文的全量快照，直接覆盖字段即可。 */
  onAnswerField: (field: AnswerTextField, text: string) => void
  onComplete: (result: AnswerPipelineResult) => void
  onError: (payload: StreamErrorPayload) => void
}

/** 与 /v1/query 结果一致，但先推思考过程、再逐段推正文。 */
export function streamAnswer(
  payload: QueryRequest,
  handlers: AnswerStreamHandlers,
): Promise<void> {
  return streamEventRequest<AnswerPipelineResult>(
    '/v1/query/stream',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(payload),
    },
    {
      onProgress: handlers.onProgress,
      onEvent(event, data) {
        if (event === 'reasoning') {
          const { text } = data as { text?: unknown }
          if (typeof text === 'string') {
            handlers.onReasoning(text)
          }
          return
        }

        if (event === 'answer') {
          const { field, text } = data as { field?: unknown; text?: unknown }
          if (
            (field === 'summary' || field === 'analysis') &&
            typeof text === 'string'
          ) {
            handlers.onAnswerField(field, text)
          }
        }
      },
      onComplete: handlers.onComplete,
      onError: handlers.onError,
    },
  )
}

export function uploadLaw(
  file: File,
  lawId: string,
  lawName: string,
  handlers: KnowledgeStreamHandlers<UploadResult>,
): Promise<void> {
  const body = new FormData()
  body.append('file', file)
  body.append('law_id', lawId)
  body.append('law_name', lawName)
  return streamKnowledgeRequest('/v1/laws/upload', body, handlers)
}

export function rebuildKnowledgeBase(
  handlers: KnowledgeStreamHandlers<RebuildResult>,
): Promise<void> {
  return streamKnowledgeRequest(
    '/v1/knowledge-base/rebuild',
    new FormData(),
    handlers,
  )
}
