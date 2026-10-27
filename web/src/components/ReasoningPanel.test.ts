import { mount } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { describe, expect, it } from 'vitest'

import ReasoningPanel from './ReasoningPanel.vue'

function mountPanel(text: string, active: boolean) {
  return mount(ReasoningPanel, {
    props: { text, active },
    global: { plugins: [ElementPlus] },
  })
}

describe('ReasoningPanel', () => {
  // 展开状态用 aria-expanded 断言，避免依赖过渡动画的时序。
  it('思考时自动展开并跟随内容', () => {
    const wrapper = mountPanel('先看证据', true)

    expect(wrapper.find('.reasoning-label').text()).toBe('正在思考…')
    expect(wrapper.find('.reasoning-toggle').attributes('aria-expanded')).toBe(
      'true',
    )
    expect(wrapper.find('.reasoning-meta').exists()).toBe(false)
  })

  it('思考结束后自动收起，点开还能看', async () => {
    const wrapper = mountPanel('先看证据', true)
    await wrapper.setProps({ active: false })

    expect(wrapper.find('.reasoning-label').text()).toBe('思考过程')
    expect(wrapper.find('.reasoning-meta').text()).toBe('4 字')
    expect(wrapper.find('.reasoning-toggle').attributes('aria-expanded')).toBe(
      'false',
    )

    await wrapper.find('.reasoning-toggle').trigger('click')
    expect(wrapper.find('.reasoning-toggle').attributes('aria-expanded')).toBe(
      'true',
    )
    expect(wrapper.find('.reasoning-text').text()).toBe('先看证据')
  })
})
