<script setup lang="ts">
import { EXAMPLE_QUESTIONS } from '@/lib/presentation'

defineProps<{ lawCount: number; disabled?: boolean }>()
const emit = defineEmits<{ send: [query: string] }>()
</script>

<template>
  <section class="hero">
    <div class="brand-mark">Evidence-first legal search</div>
    <h1>把生活问题，落到具体法条。</h1>
    <p class="hero-lead">
      描述你的情况，系统会从本地法律知识库检索依据、生成说明，
      并只展示通过校验的引用。
    </p>
    <div class="scope-strip">
      <span class="scope-chip">{{ lawCount }} 部法律</span>
      <span class="scope-chip">支持多轮追问</span>
      <span class="scope-chip">引用可追溯</span>
      <span class="scope-chip">证据不足时拒答</span>
      <span class="scope-chip">不保存完整检索正文</span>
    </div>

    <h2 class="section-kicker">试着这样问</h2>
    <div class="example-grid">
      <button
        v-for="question in EXAMPLE_QUESTIONS"
        :key="question"
        type="button"
        class="example-card"
        :disabled="disabled"
        @click="emit('send', question)"
      >
        {{ question }}
      </button>
    </div>

    <p class="empty-hint">
      也可以直接在下方输入真实情况。请避免填写身份证号、银行卡号等敏感信息。
    </p>
  </section>
</template>
