<script setup lang="ts">
import { computed } from 'vue'

import {
  citationLocation,
  resultMetrics,
  statusPresentation,
} from '@/lib/presentation'
import type { AnswerPipelineResult } from '@/types'
import FoldSection from './FoldSection.vue'

const props = defineProps<{ result: AnswerPipelineResult }>()

const answer = computed(() => props.result.answer)
const status = computed(() => statusPresentation(answer.value.status))
const metrics = computed(() => resultMetrics(props.result))
</script>

<template>
  <article class="answer-card">
    <div class="status-row">
      <span class="status-pill" :class="status.tone">{{ status.label }}</span>
      <span class="status-description">{{ status.description }}</span>
    </div>

    <p class="answer-summary">{{ answer.summary }}</p>

    <div class="metric-row">
      <div v-for="metric in metrics" :key="metric.label" class="metric">
        <span class="metric-value">{{ metric.value }}</span>
        <span class="metric-label">{{ metric.label }}</span>
      </div>
    </div>

    <div v-if="answer.follow_up_question" class="follow-up">
      <strong>还需要确认一件事</strong>
      <p>{{ answer.follow_up_question }}</p>
    </div>

    <template v-if="answer.analysis">
      <h3 class="section-kicker">分析说明</h3>
      <p class="answer-text">{{ answer.analysis }}</p>
    </template>

    <template v-if="answer.conditions.length > 0">
      <h3 class="section-kicker">适用条件</h3>
      <ul class="condition-list">
        <li v-for="(condition, index) in answer.conditions" :key="index">
          {{ condition }}
        </li>
      </ul>
    </template>

    <FoldSection
      v-if="answer.citations.length > 0"
      class="citations-fold"
      label="法条依据"
      :meta="`${answer.citations.length} 条`"
    >
      <div class="citation-list">
        <article
          v-for="(citation, index) in answer.citations"
          :key="citation.evidence_id"
          class="citation-item"
        >
          <p class="citation-item-title">
            {{ String(index + 1).padStart(2, '0') }} ·
            {{ citation.law_name }} · {{ citation.article_no }}
          </p>
          <p class="citation-meta">
            证据 {{ citation.evidence_id }} ·
            {{ citationLocation(citation) }}
          </p>
          <blockquote class="citation-quote">
            {{ citation.quote }}
          </blockquote>
        </article>
      </div>
    </FoldSection>

    <FoldSection
      v-if="answer.limitations"
      class="limits-fold"
      label="边界与限制"
    >
      <p class="answer-text">{{ answer.limitations }}</p>
    </FoldSection>

    <p class="disclaimer">
      引用经过程序校验，但法律适用仍取决于完整事实、有效版本和司法实践。
      如涉及重大权益，请咨询执业律师。
    </p>
  </article>
</template>
