<script setup lang="ts">
import { CircleCheck, CircleClose, Loading } from '@element-plus/icons-vue'

import type { ProgressEvent } from '@/types'

defineProps<{
  events: ProgressEvent[]
  state: 'running' | 'complete' | 'error'
}>()
</script>

<template>
  <div class="progress-panel">
    <div class="progress-head">
      <el-icon v-if="state === 'running'" class="is-loading">
        <Loading />
      </el-icon>
      <el-icon v-else-if="state === 'complete'" class="progress-ok">
        <CircleCheck />
      </el-icon>
      <el-icon v-else class="progress-danger">
        <CircleClose />
      </el-icon>
      <span>
        {{
          state === 'running'
            ? '正在处理，请保持页面打开'
            : state === 'complete'
              ? '处理完成'
              : '处理没有完成'
        }}
      </span>
    </div>

    <el-progress
      v-if="state === 'running'"
      :percentage="100"
      :indeterminate="true"
      :duration="2"
      :show-text="false"
      :stroke-width="4"
    />

    <ul v-if="events.length" class="progress-list">
      <li v-for="(event, index) in events" :key="`${event.stage}-${index}`">
        <span class="progress-stage">{{ event.stage }}</span>
        <span>{{ event.message }}</span>
      </li>
    </ul>
  </div>
</template>
