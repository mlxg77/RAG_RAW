import { computed, ref } from 'vue'

import { ApiError, fetchHealth, fetchLaws } from '@/api/client'
import type { HealthResponse, LawOption } from '@/types'

const CHECK_LABELS: Record<string, string> = {
  manifest: '法律清单',
  articles: '结构化法条',
  bm25_index: '关键词索引',
  chroma_index: '向量索引',
}

export function useKnowledgeBase() {
  const laws = ref<LawOption[]>([])
  const health = ref<HealthResponse | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function refresh(): Promise<void> {
    loading.value = true
    error.value = null

    try {
      const [lawList, healthResult] = await Promise.all([
        fetchLaws(),
        fetchHealth().catch(() => null),
      ])
      laws.value = lawList
      health.value = healthResult
    } catch (cause) {
      error.value =
        cause instanceof ApiError
          ? cause.message
          : '暂时读不到法律清单，请稍后刷新。'
    } finally {
      loading.value = false
    }
  }

  const ready = computed(() => health.value?.status === 'ok')

  const checks = computed(() =>
    Object.entries(health.value?.checks ?? {}).map(([key, ok]) => ({
      key,
      label: CHECK_LABELS[key] ?? key,
      ok,
    })),
  )

  return { laws, health, loading, error, ready, checks, refresh }
}
