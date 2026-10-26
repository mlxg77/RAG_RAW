<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import { rebuildKnowledgeBase } from '@/api/stream'
import type { ProgressEvent, RebuildResult } from '@/types'
import ProgressSteps from './ProgressSteps.vue'

const visible = defineModel<boolean>({ default: false })
const emit = defineEmits<{ completed: [result: RebuildResult] }>()

const events = ref<ProgressEvent[]>([])
const state = ref<'idle' | 'running' | 'complete' | 'error'>('idle')
const result = ref<RebuildResult | null>(null)
const errorMessage = ref('')
const errorDetail = ref('')

const busy = computed(() => state.value === 'running')
const successMessage = computed(() => {
  if (!result.value) {
    return ''
  }
  return (
    `知识库重建完成：${result.value.law_count} 部法律、` +
    `${result.value.chunk_total} 个 chunk。`
  )
})

watch(visible, (open) => {
  if (open) {
    reset()
  }
})

function reset(): void {
  events.value = []
  result.value = null
  errorMessage.value = ''
  errorDetail.value = ''
  state.value = 'idle'
}

function handleBeforeClose(done: () => void): void {
  if (!busy.value) {
    done()
  }
}

async function start(): Promise<void> {
  if (busy.value) {
    return
  }

  events.value = []
  result.value = null
  errorMessage.value = ''
  errorDetail.value = ''
  state.value = 'running'

  await rebuildKnowledgeBase({
    onProgress: (event) => events.value.push(event),
    onComplete: (payload) => {
      result.value = payload
      state.value = 'complete'
      emit('completed', payload)
    },
    onError: (payload) => {
      state.value = 'error'
      errorMessage.value = payload.message
      errorDetail.value = payload.detail ?? ''
    },
  })
}
</script>

<template>
  <el-dialog
    v-model="visible"
    title="重新构建现有知识库"
    width="580px"
    :close-on-click-modal="false"
    :close-on-press-escape="!busy"
    :before-close="handleBeforeClose"
  >
    <p class="dialog-lead">
      根据 manifest.yaml 和 data/raw 重新生成结构化语料、BM25 与 Chroma。
      该操作会调用 Embedding 服务，法律较多时可能需要几分钟。
    </p>

    <el-alert
      type="warning"
      title="重建期间暂时无法提问，完成或失败后会自动加载最新索引。"
      :closable="false"
      show-icon
      class="dialog-alert"
    />

    <ProgressSteps
      v-if="state !== 'idle'"
      :events="events"
      :state="state"
    />

    <el-alert
      v-if="successMessage"
      type="success"
      :title="successMessage"
      :closable="false"
      show-icon
      class="dialog-alert"
    />

    <el-alert
      v-if="errorMessage"
      type="error"
      :title="errorMessage"
      :closable="false"
      show-icon
      class="dialog-alert"
    />

    <el-collapse v-if="errorDetail" class="detail-collapse">
      <el-collapse-item title="技术信息" name="detail">
        <p class="detail-text">{{ errorDetail }}</p>
      </el-collapse-item>
    </el-collapse>

    <template #footer>
      <el-button :disabled="busy" @click="visible = false">关闭</el-button>
      <el-button
        type="primary"
        :loading="busy"
        :disabled="state === 'complete'"
        @click="start"
      >
        {{ busy ? '正在重建…' : '开始重建' }}
      </el-button>
    </template>
  </el-dialog>
</template>
