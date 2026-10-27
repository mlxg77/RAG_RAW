import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { AnswerStreamHandlers } from '@/api/stream'
import { streamAnswer } from '@/api/stream'
import type { AnswerPipelineResult } from '@/types'

vi.mock('@/api/client', () => ({
  ApiError: class ApiError extends Error {},
  createConversation: vi.fn(async () => ({ id: 7 })),
  deleteConversation: vi.fn(async () => undefined),
  fetchConversation: vi.fn(),
  fetchConversations: vi.fn(async () => []),
}))

vi.mock('@/api/stream', () => ({
  streamAnswer: vi.fn(),
}))

import { useChat } from './useChat'

const streamMock = vi.mocked(streamAnswer)

function answerResult(): AnswerPipelineResult {
  // 这里只关心消息状态的切换，不校验结果内容。
  return { query: '问题', reasoning: '想完了' } as AnswerPipelineResult
}

interface Handlers {
  current: AnswerStreamHandlers | null
}

/** 发一条消息，并把流式回调交给用例驱动。 */
async function startSend(chat: ReturnType<typeof useChat>, box: Handlers) {
  streamMock.mockImplementation(async (_payload, handlers) => {
    box.current = handlers
  })
  const pending = chat.send('公司没和我签合同怎么办？', null)
  await vi.waitFor(() => {
    expect(streamMock).toHaveBeenCalled()
  })
  return pending
}

beforeEach(() => {
  try {
    window.localStorage.removeItem('rag_law_conversation_id')
  } catch {
    // 环境不提供 localStorage 时，会话 id 本来也存不下来。
  }
  vi.clearAllMocks()
})

describe('useChat 流式问答', () => {
  it('逐段更新草稿，定稿后换成完整结果', async () => {
    const chat = useChat()
    const box: Handlers = { current: null }
    const pending = await startSend(chat, box)
    const handlers = box.current

    expect(handlers).not.toBeNull()
    expect(chat.messages.value[1].draft?.phase).toBe('thinking')

    handlers?.onProgress({ stage: 'retrieval', message: '正在检索法条' })
    expect(chat.messages.value[1].draft?.steps).toHaveLength(1)

    handlers?.onReasoning('先看证据')
    handlers?.onReasoning('再作答')
    expect(chat.messages.value[1].draft?.reasoning).toBe('先看证据再作答')

    handlers?.onAnswerField('summary', '应当订立书面')
    handlers?.onAnswerField('summary', '应当订立书面劳动合同。')
    const draft = chat.messages.value[1].draft
    expect(draft?.phase).toBe('answering')
    // 正文推的是全量快照，后到的直接覆盖前面的。
    expect(draft?.summary).toBe('应当订立书面劳动合同。')

    const result = answerResult()
    handlers?.onComplete(result)
    await pending

    const message = chat.messages.value[1]
    // ref 里的对象会被包成响应式代理，比较内容而不是引用。
    expect(message.result).toEqual(result)
    expect(message.draft).toBeUndefined()
    expect(chat.sending.value).toBe(false)
  })

  it('失败时留住已经推出来的思考过程', async () => {
    const chat = useChat()
    const box: Handlers = { current: null }
    const pending = await startSend(chat, box)

    box.current?.onReasoning('先看证据')
    box.current?.onError({ message: '这次等得太久了，问题缩短一点再试试。' })
    await pending

    const message = chat.messages.value[1]
    expect(message.error).toBe(true)
    expect(message.content).toBe('这次等得太久了，问题缩短一点再试试。')
    expect(message.result).toBeUndefined()
    expect(message.draft?.phase).toBe('failed')
    expect(message.draft?.reasoning).toBe('先看证据')
  })
})
