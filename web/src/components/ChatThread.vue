<script setup lang="ts">
import { Loading } from '@element-plus/icons-vue'

import type { ChatMessage } from '@/types'
import AnswerCard from './AnswerCard.vue'

defineProps<{ messages: ChatMessage[] }>()
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
        <div v-if="message.pending" class="thinking">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>正在检索法条并核对引用，通常需要十几秒…</span>
        </div>
        <AnswerCard v-else-if="message.result" :result="message.result" />
        <p v-else class="thread-error">{{ message.content }}</p>
      </div>
    </div>
  </div>
</template>
