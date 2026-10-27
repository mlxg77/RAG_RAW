import { afterEach, describe, expect, it, vi } from 'vitest'

import type { AnswerPipelineResult, QueryRequest } from '@/types'

import { streamAnswer } from './stream'

const PAYLOAD: QueryRequest = {
  query: '公司没和我签合同怎么办？',
  law_name: null,
  history: [],
  conversation_id: null,
}

interface Frame {
  event: string
  data: unknown
}

/** 造一个只够 stream.ts 读的响应：ok / status / body 三件套。 */
function sseResponse(frames: Frame[]): unknown {
  const encoder = new TextEncoder()
  const text = frames
    .map((frame) => {
      const data = JSON.stringify(frame.data)
      return `event: ${frame.event}\ndata: ${data}\n\n`
    })
    .join('')

  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(text))
      controller.close()
    },
  })

  return { ok: true, status: 200, body }
}

function stubFetch(response: unknown): ReturnType<typeof vi.fn> {
  const mock = vi.fn(async () => response)
  vi.stubGlobal('fetch', mock)
  return mock
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('streamAnswer', () => {
  it('把思考过程、正文快照和最终结果分别交给调用方', async () => {
    const result = { query: '问题', reasoning: '想完了' } as AnswerPipelineResult
    const fetcher = stubFetch(
      sseResponse([
        { event: 'progress', data: { stage: 'retrieval', message: '正在检索法条' } },
        { event: 'reasoning', data: { text: '先看证据' } },
        { event: 'reasoning', data: { text: '再下结论' } },
        { event: 'answer', data: { field: 'summary', text: '应当订立书面合同。' } },
        { event: 'answer', data: { field: 'analysis', text: '证据只说明了形式要求。' } },
        { event: 'complete', data: result },
      ]),
    )

    const stages: string[] = []
    const reasoning: string[] = []
    const fields: Array<[string, string]> = []
    let completed: AnswerPipelineResult | null = null

    await streamAnswer(PAYLOAD, {
      onProgress: (event) => stages.push(event.stage),
      onReasoning: (text) => reasoning.push(text),
      onAnswerField: (field, text) => fields.push([field, text]),
      onComplete: (value) => {
        completed = value
      },
      onError: (payload) => {
        throw new Error(payload.message)
      },
    })

    expect(stages).toEqual(['retrieval'])
    expect(reasoning).toEqual(['先看证据', '再下结论'])
    expect(fields).toEqual([
      ['summary', '应当订立书面合同。'],
      ['analysis', '证据只说明了形式要求。'],
    ])
    expect(completed).toEqual(result)

    const [url, init] = fetcher.mock.calls[0] as [string, RequestInit]
    expect(url).toContain('/v1/query/stream')
    expect(init.method).toBe('POST')
    expect(init.body).toBe(JSON.stringify(PAYLOAD))
  })

  it('忽略认不出来的字段名，不污染草稿', async () => {
    stubFetch(
      sseResponse([
        { event: 'answer', data: { field: 'citations', text: '[]' } },
        { event: 'answer', data: { field: 'summary', text: '结论' } },
        { event: 'complete', data: { query: '问题' } },
      ]),
    )

    const fields: Array<[string, string]> = []
    await streamAnswer(PAYLOAD, {
      onProgress: () => undefined,
      onReasoning: () => undefined,
      onAnswerField: (field, text) => fields.push([field, text]),
      onComplete: () => undefined,
      onError: () => undefined,
    })

    expect(fields).toEqual([['summary', '结论']])
  })

  it('把服务端 error 帧转成提示文案', async () => {
    stubFetch(
      sseResponse([
        { event: 'error', data: { message: '这次等得太久了，问题缩短一点再试试。' } },
      ]),
    )

    const messages: string[] = []
    await streamAnswer(PAYLOAD, {
      onProgress: () => undefined,
      onReasoning: () => undefined,
      onAnswerField: () => undefined,
      onComplete: () => undefined,
      onError: (payload) => messages.push(payload.message),
    })

    expect(messages).toEqual(['这次等得太久了，问题缩短一点再试试。'])
  })
})
