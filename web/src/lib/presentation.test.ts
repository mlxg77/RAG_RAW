import { describe, expect, it } from 'vitest'

import type {
  AnswerPipelineResult,
  ChatMessage,
  ValidatedCitation,
} from '@/types'

import {
  buildConversationHistory,
  citationLocation,
  formatConversationTime,
  friendlyErrorMessage,
  resultMetrics,
  statusPresentation,
} from './presentation'

function citation(
  overrides: Partial<ValidatedCitation> = {},
): ValidatedCitation {
  return {
    evidence_id: 'E001',
    chunk_id: 'law:1',
    law_id: 'law',
    law_name: '中华人民共和国测试法',
    article_no: '第一条',
    quote: '这是经过校验的法条原文。',
    source_file: 'data/raw/test.pdf',
    page: 2,
    end_page: 3,
    ...overrides,
  }
}

function result(): AnswerPipelineResult {
  return {
    query: '测试问题',
    answer: {
      status: 'answered',
      summary: '简明结论',
      analysis: '分析',
      conditions: ['条件一'],
      citations: [citation()],
      limitations: '限制',
      follow_up_question: null,
    },
    retrieval: {
      query: '测试问题',
      reranked_results: [],
      context: {
        text: '',
        evidences: [],
        estimated_tokens: 0,
        max_tokens: 100,
        truncated: false,
      },
      retrieval_latency_ms: 100,
      context_latency_ms: 10,
      total_latency_ms: 110,
    },
    generation_attempts: 1,
    top_relevance_score: 0.9,
    validation_errors: [],
    prompt_version: 'test-v1',
    generation_latency_ms: 1000,
    total_latency_ms: 2500,
  }
}

function userMessage(content: string, id = content): ChatMessage {
  return { id, role: 'user', content }
}

function assistantMessage(id = 'assistant'): ChatMessage {
  return { id, role: 'assistant', result: result() }
}

describe('citationLocation', () => {
  it('formats page ranges and single pages', () => {
    expect(citationLocation(citation())).toBe(
      'data/raw/test.pdf · 第 2–3 页',
    )
    expect(citationLocation(citation({ end_page: 2 }))).toBe(
      'data/raw/test.pdf · 第 2 页',
    )
    expect(
      citationLocation(citation({ page: null, end_page: null })),
    ).toBe('data/raw/test.pdf')
  })
})

describe('statusPresentation', () => {
  it('keeps the user facing labels and tones', () => {
    expect(statusPresentation('answered')).toMatchObject({
      label: '已找到依据',
      tone: 'success',
    })
    expect(statusPresentation('generation_failed').tone).toBe('danger')
  })
})

describe('resultMetrics', () => {
  it('summarises citations, latency and attempts', () => {
    expect(resultMetrics(result())).toEqual([
      { label: '引用', value: '1' },
      { label: '总耗时', value: '2.5 秒' },
      { label: '生成次数', value: '1' },
    ])
  })
})

describe('buildConversationHistory', () => {
  it('keeps user text and assistant conclusions only', () => {
    const messages: ChatMessage[] = [
      userMessage('公司没有和我签合同。'),
      assistantMessage('a1'),
      { id: 'error', role: 'assistant', content: '这次没能完成', error: true },
      userMessage('那赔偿怎么算？', 'u2'),
      { id: 'pending', role: 'assistant', pending: true },
    ]

    const history = buildConversationHistory(messages)

    expect(history).toEqual([
      { role: 'user', content: '公司没有和我签合同。' },
      {
        role: 'assistant',
        content: '简明结论\n分析\n适用条件：条件一',
      },
      { role: 'user', content: '那赔偿怎么算？' },
    ])
  })

  it('caps history at 50 turns and starts with a user turn', () => {
    const messages: ChatMessage[] = []
    for (let index = 0; index < 25; index += 1) {
      messages.push(userMessage(`问题 ${index}`, `u${index}`))
      messages.push(assistantMessage(`a${index}`))
    }
    messages.push(userMessage('最后的问题', 'u-last'))

    const history = buildConversationHistory(messages)

    expect(history).toHaveLength(49)
    expect(history[0]?.role).toBe('user')
    expect(history.at(-1)).toEqual({
      role: 'user',
      content: '最后的问题',
    })
  })
})

describe('formatConversationTime', () => {
  const now = new Date('2026-10-26T20:00:00+08:00')

  it('shows the clock for today and the date for earlier days', () => {
    expect(
      formatConversationTime('2026-10-26T09:30:00+08:00', now),
    ).toContain('09')
    expect(
      formatConversationTime('2026-10-20T09:30:00+08:00', now),
    ).toContain('20')
  })

  it('returns an empty string for unusable timestamps', () => {
    expect(formatConversationTime('not-a-date', now)).toBe('')
  })
})

describe('friendlyErrorMessage', () => {
  it('maps common failures to plain language', () => {
    expect(friendlyErrorMessage(429)).toContain('歇一分钟')
    expect(friendlyErrorMessage(504)).toContain('等得太久')
    expect(friendlyErrorMessage(503)).toContain('还没准备好')
    expect(friendlyErrorMessage(422, '指定法律不在当前知识库中')).toBe(
      '指定法律不在当前知识库中',
    )
    expect(friendlyErrorMessage(500)).toBe('这次没能完成，请稍后重试。')
  })
})
