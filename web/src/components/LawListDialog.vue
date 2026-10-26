<script setup lang="ts">
import type { LawOption } from '@/types'

const visible = defineModel<boolean>({ default: false })

defineProps<{ laws: LawOption[] }>()
</script>

<template>
  <el-dialog v-model="visible" title="已收录的法律" width="540px">
    <p v-if="laws.length === 0" class="dialog-empty">
      还没有收录法律，先添加一部吧。
    </p>

    <ol v-else class="law-list">
      <li v-for="(law, index) in laws" :key="law.law_id">
        <span class="law-index">{{ String(index + 1).padStart(2, '0') }}</span>
        <span class="law-name">{{ law.law_name }}</span>
        <span class="law-id">{{ law.law_id }}</span>
      </li>
    </ol>

    <template #footer>
      <el-button type="primary" @click="visible = false">知道了</el-button>
    </template>
  </el-dialog>
</template>
