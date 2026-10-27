<script setup lang="ts">
import { ArrowDown } from '@element-plus/icons-vue'
import { ref } from 'vue'

defineProps<{
  label: string
  /** 按钮右侧的次要信息，例如「3 条」。 */
  meta?: string
}>()

/** 一律默认收起，想看再把内容铺开。 */
const open = ref(false)
</script>

<template>
  <section class="fold-section">
    <button
      class="fold-toggle"
      type="button"
      :aria-expanded="open"
      @click="open = !open"
    >
      <span>{{ label }}</span>
      <span v-if="meta" class="fold-meta">{{ meta }}</span>
      <el-icon class="fold-chevron" :class="{ 'is-open': open }">
        <ArrowDown />
      </el-icon>
    </button>

    <el-collapse-transition>
      <div v-show="open" class="fold-body">
        <slot />
      </div>
    </el-collapse-transition>
  </section>
</template>
