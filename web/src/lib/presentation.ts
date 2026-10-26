import type {
  AnswerPipelineResult,
  AnswerStatus,
  ChatMessage,
  ConversationTurn,
  ValidatedCitation,
} from '@/types'

export interface StatusPresentation {
  label: string
  tone: 'success' | 'warning' | 'neutral' | 'danger'
  description: string
}

export interface Metric {
  label: string
  value: string
}

const MAX_HISTORY_TURNS = 50
const MAX_HISTORY_CONTENT = 2000

export const STATUS_PRESENTATIONS: Record<AnswerStatus, StatusPresentation> = {
  answered: {
    label: '已找到依据',
    tone: 'success',
    description: '回答中的引用已经过程序校验。',
  },
  needs_clarification: {
    label: '需要补充信息',
    tone: 'warning',
    description: '补充一个关键事实后，回答会更准确。',
  },
  insufficient_evidence: {
    label: '依据不足',
    tone: 'neutral',
    description: '当前知识库没有提供足够证据。',
  },
  generation_failed: {
    label: '已安全停止',
    tone: 'danger',
    description: '本次生成或引用校验未通过，未展示未经验证的内容。',
  },
}

export const EXAMPLE_QUESTIONS = [
  '公司一直没和我签书面劳动合同，我可以要求什么？',
  '酒后驾驶机动车会受到什么处罚？',
  '消费者买到不符合食品安全标准的食品，可以要求赔偿吗？',
] as const

export function statusPresentation(status: AnswerStatus): StatusPresentation {
  return STATUS_PRESENTATIONS[status]
}

export function citationLocation(citation: ValidatedCitation): string {
  const location = citation.source_file
  if (citation.page === null) {
    return location
  }

  if (citation.end_page !== null && citation.end_page !== citation.page) {
    return `${location} · 第 ${citation.page}–${citation.end_page} 页`
  }

  return `${location} · 第 ${citation.page} 页`
}

/** 历史对话列表里的时间：今天只显示时分，更早显示日期。 */
export function formatConversationTime(
  value: string,
  now: Date = new Date(),
): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return ''
  }

  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString('zh-CN', {
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  if (date.getFullYear() === now.getFullYear()) {
    return date.toLocaleDateString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
    })
  }

  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  })
}

export function resultMetrics(result: AnswerPipelineResult): Metric[] {
  return [
    { label: '引用', value: String(result.answer.citations.length) },
    { label: '总耗时', value: `${(result.total_latency_ms / 1000).toFixed(1)} 秒` },
    { label: '生成次数', value: String(result.generation_attempts) },
  ]
}

/**
 * 把页面消息还原成 API 的历史入参。
 *
 * 助手历史只保留结论与条件，不把引用原文回灌给模型，避免形成
 * “引用自己的回答”的证据闭环；与后端历史长度上限保持一致。
 */
export function buildConversationHistory(
  messages: ChatMessage[],
): ConversationTurn[] {
  const turns: ConversationTurn[] = []

  for (const message of messages) {
    if (message.role === 'user' && message.content) {
      turns.push({ role: 'user', content: message.content })
      continue
    }

    if (message.role !== 'assistant' || !message.result) {
      continue
    }

    const answer = message.result.answer
    const parts = [answer.summary, answer.analysis]
    if (answer.conditions.length > 0) {
      parts.push(`适用条件：${answer.conditions.join('；')}`)
    }
    if (answer.follow_up_question) {
      parts.push(`追问：${answer.follow_up_question}`)
    }

    const content = parts.filter(Boolean).join('\n').trim()
    if (content) {
      turns.push({
        role: 'assistant',
        content: content.slice(0, MAX_HISTORY_CONTENT),
      })
    }
  }

  const recent = turns.slice(-MAX_HISTORY_TURNS)
  // 历史以上一轮助手回答开头会让模型误读，丢掉这一条。
  if (recent[0]?.role === 'assistant') {
    recent.shift()
  }
  return recent
}

export function friendlyErrorMessage(
  status: number,
  detail?: string,
): string {
  switch (status) {
    case 401:
      return detail?.trim() || '登录已经过期，请重新登录。'
    case 403:
      return detail?.trim() || '当前账号暂时不能使用，请联系管理员。'
    case 409:
      return '知识库正在更新，等这次任务结束再试。'
    case 413:
      return '文件太大了，请控制在 50 MB 以内。'
    case 422:
      return detail?.trim() || '提交的内容不符合要求，请检查后重试。'
    case 429:
      return '提问有点快，歇一分钟再试。'
    case 503:
      return '法律资料还没准备好，请稍后再试。'
    case 504:
      return '这次等得太久了，问题缩短一点再试试。'
    default:
      return detail?.trim() || '这次没能完成，请稍后重试。'
  }
}
