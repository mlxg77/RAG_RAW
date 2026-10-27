import { mount } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { describe, expect, it } from 'vitest'

import type { AnswerPipelineResult } from '@/types'

import AnswerCard from './AnswerCard.vue'

function result(): AnswerPipelineResult {
  return {
    query: '公司没和我签合同怎么办？',
    answer: {
      status: 'answered',
      summary: '建立劳动关系应当订立书面劳动合同。',
      analysis: '检索到的证据只说明了书面形式要求。',
      conditions: ['需要确认双方是否已经建立劳动关系。'],
      citations: [
        {
          evidence_id: 'E001',
          chunk_id: 'law:10',
          law_id: 'labor-contract-law',
          law_name: '中华人民共和国劳动合同法',
          article_no: '第十条',
          quote: '建立劳动关系，应当订立书面劳动合同。',
          source_file: 'data/raw/劳动合同法.pdf',
          page: 3,
          end_page: null,
        },
      ],
      limitations: '回答仅基于当前知识库。',
      follow_up_question: null,
    },
    retrieval: {
      query: '公司没和我签合同怎么办？',
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
    prompt_version: 'stage7-v1-json-stream',
    generation_latency_ms: 1000,
    total_latency_ms: 2500,
    reasoning: '先确认劳动关系，再找书面形式的规定。',
  }
}

function mountCard() {
  return mount(AnswerCard, {
    props: { result: result() },
    global: { plugins: [ElementPlus] },
  })
}

describe('AnswerCard', () => {
  it('法条依据默认收起，只露一个按钮', () => {
    const wrapper = mountCard()
    const toggle = wrapper.find('.citations-fold .fold-toggle')

    expect(toggle.text()).toContain('法条依据')
    expect(toggle.text()).toContain('1 条')
    expect(toggle.attributes('aria-expanded')).toBe('false')
  })

  it('点一下才展开法条原文', async () => {
    const wrapper = mountCard()
    const section = wrapper.find('.citations-fold')

    await section.find('.fold-toggle').trigger('click')

    expect(section.find('.fold-toggle').attributes('aria-expanded')).toBe(
      'true',
    )
    expect(section.find('.citation-item-title').text()).toContain('第十条')
    expect(section.find('.citation-quote').text()).toBe(
      '建立劳动关系，应当订立书面劳动合同。',
    )
  })

  it('边界与限制用同一种按钮，也是点开才看', async () => {
    const wrapper = mountCard()
    const section = wrapper.find('.limits-fold')
    const toggle = section.find('.fold-toggle')

    expect(toggle.text()).toContain('边界与限制')
    expect(toggle.attributes('aria-expanded')).toBe('false')

    await toggle.trigger('click')

    expect(toggle.attributes('aria-expanded')).toBe('true')
    expect(section.find('.answer-text').text()).toBe('回答仅基于当前知识库。')
  })

  it('总结与分析照常显示', () => {
    const wrapper = mountCard()

    expect(wrapper.find('.answer-summary').text()).toBe(
      '建立劳动关系应当订立书面劳动合同。',
    )
    expect(wrapper.find('.answer-text').text()).toBe(
      '检索到的证据只说明了书面形式要求。',
    )
  })
})
