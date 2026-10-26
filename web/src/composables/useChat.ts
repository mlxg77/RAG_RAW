import { computed, ref } from 'vue'

import { ApiError, queryAnswer } from '@/api/client'
import { buildConversationHistory } from '@/lib/presentation'
import type { ChatMessage } from '@/types'

function createId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export function useChat() {
  const messages = ref<ChatMessage[]>([])
  const sending = ref(false)
  const hasMessages = computed(() => messages.value.length > 0)

  function replace(id: string, next: ChatMessage): void {
    const index = messages.value.findIndex((message) => message.id === id)
    if (index >= 0) {
      messages.value[index] = next
    }
  }

  async function send(query: string, lawName: string | null): Promise<void> {
    const trimmed = query.trim()
    if (!trimmed || sending.value) {
      return
    }

    const history = buildConversationHistory(messages.value)
    const pendingId = createId()

    messages.value.push({
      id: createId(),
      role: 'user',
      content: trimmed,
      lawName,
    })
    messages.value.push({ id: pendingId, role: 'assistant', pending: true })
    sending.value = true

    try {
      const result = await queryAnswer({
        query: trimmed,
        law_name: lawName,
        history,
      })
      replace(pendingId, { id: pendingId, role: 'assistant', result })
    } catch (error) {
      const message =
        error instanceof ApiError
          ? error.message
          : '这次没能完成，请稍后重试。'
      replace(pendingId, {
        id: pendingId,
        role: 'assistant',
        content: message,
        error: true,
      })
    } finally {
      sending.value = false
    }
  }

  function clear(): void {
    messages.value = []
  }

  return { messages, sending, hasMessages, send, clear }
}
