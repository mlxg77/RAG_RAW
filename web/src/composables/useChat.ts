import { computed, ref } from 'vue'

import {
  ApiError,
  createConversation,
  deleteConversation,
  fetchConversation,
  fetchConversations,
  queryAnswer,
} from '@/api/client'
import { buildConversationHistory } from '@/lib/presentation'
import type { ChatMessage, ConversationSummary } from '@/types'

/** 当前会话 id：刷新页面或服务重启后靠它把上一次的对话接回来。 */
const STORAGE_KEY = 'rag_law_conversation_id'

function createId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function readStoredConversationId(): number | null {
  try {
    const parsed = Number(window.localStorage.getItem(STORAGE_KEY))
    return Number.isInteger(parsed) && parsed > 0 ? parsed : null
  } catch {
    return null
  }
}

function storeConversationId(id: number | null): void {
  try {
    if (id === null) {
      window.localStorage.removeItem(STORAGE_KEY)
    } else {
      window.localStorage.setItem(STORAGE_KEY, String(id))
    }
  } catch {
    // 无痕模式下写不进去，退化成「只在本次打开有效」。
  }
}

export function useChat() {
  const messages = ref<ChatMessage[]>([])
  const conversations = ref<ConversationSummary[]>([])
  const conversationId = ref<number | null>(readStoredConversationId())
  const sending = ref(false)
  const loadingHistory = ref(false)
  const hasMessages = computed(() => messages.value.length > 0)

  function replace(id: string, next: ChatMessage): void {
    const index = messages.value.findIndex((message) => message.id === id)
    if (index >= 0) {
      messages.value[index] = next
    }
  }

  function resetToBlank(): void {
    messages.value = []
    conversationId.value = null
    storeConversationId(null)
  }

  async function refreshConversations(): Promise<void> {
    try {
      conversations.value = await fetchConversations()
    } catch {
      // 历史列表拉不到不影响问答，下次操作时会再试。
    }
  }

  /** 首次提问时补建会话；建不上就退化成「这一轮不记历史」。 */
  async function ensureConversation(
    lawName: string | null,
  ): Promise<number | null> {
    if (conversationId.value !== null) {
      return conversationId.value
    }

    try {
      const created = await createConversation(lawName)
      conversationId.value = created.id
      storeConversationId(created.id)
      return created.id
    } catch {
      return null
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
      const targetId = await ensureConversation(lawName)
      const result = await queryAnswer({
        query: trimmed,
        law_name: lawName,
        history,
        conversation_id: targetId,
      })
      replace(pendingId, { id: pendingId, role: 'assistant', result })
      if (targetId !== null) {
        void refreshConversations()
      }
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

  /** 打开一条历史对话；拉取失败会抛出，交给界面提示。 */
  async function openConversation(id: number): Promise<void> {
    if (sending.value) {
      return
    }
    if (id === conversationId.value && messages.value.length > 0) {
      return
    }

    loadingHistory.value = true
    try {
      const detail = await fetchConversation(id)
      conversationId.value = detail.id
      storeConversationId(detail.id)
      messages.value = detail.messages.map((message) => ({
        id: `saved-${message.id}`,
        role: message.role,
        content: message.content,
        result: message.result ?? undefined,
      }))
    } finally {
      loadingHistory.value = false
    }
  }

  /** 开一条空白对话；真正的记录等第一次提问时才建。 */
  function startNew(): void {
    resetToBlank()
  }

  async function removeConversation(id: number): Promise<void> {
    await deleteConversation(id)
    if (conversationId.value === id) {
      resetToBlank()
    }
    await refreshConversations()
  }

  /** 清空当前对话，同时把它从历史里删掉。 */
  async function clearCurrent(): Promise<void> {
    const id = conversationId.value
    resetToBlank()
    if (id !== null) {
      await deleteConversation(id)
      await refreshConversations()
    }
  }

  /** 恢复上一次的对话：先拉列表，再把当前会话的消息读回来。 */
  async function restore(): Promise<void> {
    await refreshConversations()
    const stored = conversationId.value ?? readStoredConversationId()
    if (stored === null) {
      return
    }

    try {
      await openConversation(stored)
    } catch {
      // 会话已被删除或换了账号时，安静地回到空白状态。
      resetToBlank()
      await refreshConversations()
    }
  }

  /** 退出登录时清干净，避免下一个账号看到上一个人的东西。 */
  function forget(): void {
    resetToBlank()
    conversations.value = []
  }

  return {
    messages,
    conversations,
    conversationId,
    sending,
    loadingHistory,
    hasMessages,
    send,
    startNew,
    openConversation,
    removeConversation,
    clearCurrent,
    restore,
    forget,
  }
}
