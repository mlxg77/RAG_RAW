<script setup lang="ts">
import { ArrowDown } from '@element-plus/icons-vue'
import { computed, nextTick, ref, watch } from 'vue'

const props = defineProps<{
  text: string
  /** 还在思考时自动展开并跟到底部，思考结束就自动收起。 */
  active: boolean
}>()

const open = ref(props.active)
const bodyRef = ref<HTMLElement | null>(null)
const charCount = computed(() => props.text.trim().length)

async function scrollToEnd(): Promise<void> {
  await nextTick()
  const body = bodyRef.value
  if (body) {
    body.scrollTop = body.scrollHeight
  }
}

watch(
  () => props.active,
  (active) => {
    open.value = active
    if (active) {
      void scrollToEnd()
    }
  },
)

watch(
  () => props.text,
  () => {
    if (props.active) {
      void scrollToEnd()
    }
  },
)
</script>

<template>
  <section class="reasoning-panel">
    <button
      class="reasoning-toggle"
      type="button"
      :aria-expanded="open"
      @click="open = !open"
    >
      <span class="reasoning-dot" :class="{ 'is-live': active }" />
      <span class="reasoning-label">
        {{ active ? '正在思考…' : '思考过程' }}
      </span>
      <span v-if="!active" class="reasoning-meta">{{ charCount }} 字</span>
      <el-icon class="reasoning-chevron" :class="{ 'is-open': open }">
        <ArrowDown />
      </el-icon>
    </button>

    <el-collapse-transition>
      <div v-show="open" ref="bodyRef" class="reasoning-body">
        <p class="reasoning-text">{{ text }}</p>
      </div>
    </el-collapse-transition>
  </section>
</template>
