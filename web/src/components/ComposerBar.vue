<script setup lang="ts">
import { Promotion } from '@element-plus/icons-vue'
import { ref } from 'vue'

const props = defineProps<{ disabled?: boolean }>()
const emit = defineEmits<{ send: [query: string] }>()

const value = ref('')

function submit(): void {
  const query = value.value.trim()
  if (!query || props.disabled) {
    return
  }
  emit('send', query)
  value.value = ''
}

function handleKeydown(event: KeyboardEvent): void {
  if (
    event.key === 'Enter' &&
    !event.shiftKey &&
    !event.isComposing
  ) {
    event.preventDefault()
    submit()
  }
}
</script>

<template>
  <div class="composer">
    <el-input
      v-model="value"
      class="composer-input"
      type="textarea"
      :autosize="{ minRows: 1, maxRows: 5 }"
      maxlength="1000"
      show-word-limit
      resize="none"
      placeholder="描述你的法律问题…"
      :disabled="disabled"
      @keydown="handleKeydown"
    />
    <el-button
      type="primary"
      class="composer-submit"
      :icon="Promotion"
      :loading="disabled"
      :disabled="!value.trim()"
      @click="submit"
    >
      查找相关法条
    </el-button>
  </div>
  <p class="composer-hint">
    Enter 发送，Shift + Enter 换行。请勿填写身份证号、银行卡号等敏感信息。
  </p>
</template>
