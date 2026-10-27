<script setup lang="ts">
import { Loading } from '@element-plus/icons-vue'

import type { ChatMessage } from '@/types'
import AnswerCard from './AnswerCard.vue'
import AnswerDraftCard from './AnswerDraftCard.vue'
import ReasoningPanel from './ReasoningPanel.vue'

defineProps<{ messages: ChatMessage[] }>()

/** 定稿后从结果里取思考过程，生成中就取草稿里的。 */
function reasoningOf(message: ChatMessage): string {
  return message.result?.reasoning ?? message.draft?.reasoning ?? ''
}

function isThinking(message: ChatMessage): boolean {
  return message.draft?.phase === 'thinking'
}
</script>

<template>
  <div class="thread" aria-live="polite">
    <div
      v-for="message in messages"
      :key="message.id"
      class="thread-item"
      :class="message.role"
    >
      <div v-if="message.role === 'user'" class="user-bubble">
        <p class="user-text">{{ message.content }}</p>
        <span v-if="message.lawName" class="message-scope">
          检索范围：{{ message.lawName }}
        </span>
      </div>

      <div v-else class="assistant-block">
        <ReasoningPanel
          v-if="reasoningOf(message)"
          :text="reasoningOf(message)"
          :active="isThinking(message)"
        />

        <AnswerDraftCard
          v-if="message.draft && message.draft.phase !== 'failed'"
          :draft="message.draft"
        />
        <AnswerCard v-else-if="message.result" :result="message.result" />
        <p v-else-if="message.error" class="thread-error">
          {{ message.content }}
        </p>
        <div v-else class="thinking">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>正在检索法条并核对引用，通常需要十几秒…</span>
        </div>
      </div>
    </div>
  </div>
</template>
