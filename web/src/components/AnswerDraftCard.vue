<script setup lang="ts">
import { Loading } from '@element-plus/icons-vue'
import { computed } from 'vue'

import type { AnswerDraft } from '@/types'

const props = defineProps<{ draft: AnswerDraft }>()

const phaseLabel = computed(() =>
  props.draft.phase === 'answering' ? '正在写回答' : '正在梳理依据',
)
const lastStep = computed(
  () => props.draft.steps[props.draft.steps.length - 1],
)
/** 光标跟着最后一段有内容的正文走。 */
const caretInAnalysis = computed(() => props.draft.analysis.length > 0)
</script>

<template>
  <article class="answer-card draft-card">
    <div class="status-row">
      <span class="status-pill draft">
        <el-icon class="is-loading"><Loading /></el-icon>
        {{ phaseLabel }}
      </span>
      <span class="status-description">先看内容，引用核对完才算定稿。</span>
    </div>

    <p v-if="draft.summary" class="answer-summary">
      {{ draft.summary }}<span v-if="!caretInAnalysis" class="typing-caret" />
    </p>
    <p v-else class="draft-placeholder">正在组织回答…</p>

    <template v-if="draft.analysis">
      <h3 class="section-kicker">分析说明</h3>
      <p class="answer-text">{{ draft.analysis }}<span class="typing-caret" /></p>
    </template>

    <p v-if="lastStep" class="draft-step">{{ lastStep.message }}</p>
  </article>
</template>
