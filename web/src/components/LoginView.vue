<script setup lang="ts">
import { Lock, User } from '@element-plus/icons-vue'
import { ref } from 'vue'

import { useAuth } from '@/composables/useAuth'

const { signIn, submitting, error, clearError } = useAuth()

const username = ref('')
const password = ref('')

async function handleSubmit(): Promise<void> {
  const name = username.value.trim()
  if (!name || !password.value || submitting.value) {
    return
  }
  await signIn({ username: name, password: password.value })
}
</script>

<template>
  <div class="login-shell">
    <div class="login-card">
      <div class="brand-lockup">
        <div class="brand-mark">RAG Law</div>
        <p>证据优先的法律问答</p>
      </div>

      <h1 class="login-title">登录后开始提问</h1>
      <p class="login-lead">
        请使用管理员分配的账号登录。忘记密码或账号被停用时，请联系系统管理员。
      </p>

      <div class="login-form">
        <el-input
          v-model="username"
          class="login-field"
          :prefix-icon="User"
          placeholder="账号"
          size="large"
          autocomplete="username"
          @input="clearError"
        />
        <el-input
          v-model="password"
          class="login-field"
          :prefix-icon="Lock"
          type="password"
          placeholder="密码"
          size="large"
          show-password
          autocomplete="current-password"
          @input="clearError"
          @keyup.enter="handleSubmit"
        />

        <el-alert
          v-if="error"
          :title="error"
          type="error"
          :closable="false"
          show-icon
          class="login-alert"
        />

        <el-button
          class="login-submit"
          type="primary"
          size="large"
          :loading="submitting"
          @click="handleSubmit"
        >
          登录
        </el-button>
      </div>

      <p class="login-foot">
        回答仅供法律信息检索与参考，不构成法律意见。
      </p>
    </div>
  </div>
</template>
