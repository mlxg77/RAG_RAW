<script setup lang="ts">
import { Delete, Document, Plus, Refresh } from '@element-plus/icons-vue'
import { computed } from 'vue'

import type { HealthResponse, LawOption } from '@/types'

const props = defineProps<{
  laws: LawOption[]
  selectedLaw: string | null
  health: HealthResponse | null
  ready: boolean
  checks: Array<{ key: string; label: string; ok: boolean }>
  loading: boolean
  hasMessages: boolean
}>()

const emit = defineEmits<{
  'update:selectedLaw': [value: string | null]
  clear: []
  openLawList: []
  openUpload: []
  openRebuild: []
}>()

const healthLabel = computed(() => {
  if (!props.health) {
    return '未连接'
  }
  return props.ready ? '索引就绪' : '索引不完整'
})

const healthTagType = computed<'success' | 'warning' | 'info'>(() => {
  if (!props.health) {
    return 'info'
  }
  return props.ready ? 'success' : 'warning'
})

function handleLawChange(value: string): void {
  emit('update:selectedLaw', value || null)
}
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar-scroll">
      <div class="brand-lockup">
        <div class="brand-mark">RAG Law</div>
        <p>证据优先的法律问答</p>
      </div>

      <section class="sidebar-card">
        <h2>检索范围</h2>
        <el-select
          :model-value="selectedLaw ?? ''"
          class="law-select"
          placeholder="全部法律"
          :loading="loading"
          @change="handleLawChange"
        >
          <el-option label="全部法律" value="" />
          <el-option
            v-for="law in laws"
            :key="law.law_id"
            :label="law.law_name"
            :value="law.law_name"
          />
        </el-select>
        <p class="hint">指定后只在这一部法律中检索。</p>
      </section>

      <section class="sidebar-card">
        <div class="card-head">
          <h2>当前知识库</h2>
          <el-tag size="small" :type="healthTagType" effect="plain">
            {{ healthLabel }}
          </el-tag>
        </div>

        <p class="kb-count">
          <strong>{{ laws.length }}</strong> 部法律
          <span>原始文本以本地知识库为准</span>
        </p>

        <ul v-if="checks.length" class="check-list">
          <li
            v-for="check in checks"
            :key="check.key"
            :class="{ 'is-ok': check.ok }"
          >
            <span class="check-dot" />
            {{ check.label }}
          </li>
        </ul>

        <div class="sidebar-actions">
          <el-button :icon="Document" @click="emit('openLawList')">
            查看法律清单
          </el-button>
          <el-button :icon="Plus" @click="emit('openUpload')">
            添加新法律
          </el-button>
          <el-button :icon="Refresh" @click="emit('openRebuild')">
            重新构建现有知识库
          </el-button>
        </div>
      </section>

      <div class="sidebar-bottom">
        <p class="disclaimer">
          回答仅供法律信息检索与参考，不构成法律意见。
        </p>
        <el-button
          class="clear-button"
          :icon="Delete"
          :disabled="!hasMessages"
          @click="emit('clear')"
        >
          清空当前对话
        </el-button>
      </div>
    </div>
  </aside>
</template>
