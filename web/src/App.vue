<script setup lang="ts">
import { Expand } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { nextTick, onMounted, ref, watch } from 'vue'

import AppSidebar from '@/components/AppSidebar.vue'
import ChatThread from '@/components/ChatThread.vue'
import ComposerBar from '@/components/ComposerBar.vue'
import HeroPanel from '@/components/HeroPanel.vue'
import LawListDialog from '@/components/LawListDialog.vue'
import LoginView from '@/components/LoginView.vue'
import RebuildDialog from '@/components/RebuildDialog.vue'
import UploadLawDialog from '@/components/UploadLawDialog.vue'
import { ApiError } from '@/api/client'
import { useAuth } from '@/composables/useAuth'
import { useChat } from '@/composables/useChat'
import { useKnowledgeBase } from '@/composables/useKnowledgeBase'
import type { RebuildResult, UploadResult } from '@/types'

const { user, ready: authReady, bootstrap, signOut } = useAuth()
const {
  messages,
  conversations,
  conversationId,
  sending,
  loadingHistory,
  hasMessages,
  send,
  startNew,
  openConversation,
  removeConversation,
  clearCurrent,
  restore,
  forget,
} = useChat()
const {
  laws,
  health,
  loading,
  error: kbError,
  ready,
  checks,
  refresh,
} = useKnowledgeBase()

const selectedLaw = ref<string | null>(null)
const sidebarOpen = ref(false)
const lawListOpen = ref(false)
const uploadOpen = ref(false)
const rebuildOpen = ref(false)
const scrollArea = ref<HTMLElement | null>(null)

watch(
  messages,
  async () => {
    await nextTick()
    const area = scrollArea.value
    if (area) {
      area.scrollTop = area.scrollHeight
    }
  },
  { deep: true },
)

watch(laws, (list) => {
  if (
    selectedLaw.value &&
    !list.some((law) => law.law_name === selectedLaw.value)
  ) {
    selectedLaw.value = null
  }
})

onMounted(bootstrap)

// 登录成功（含刷新页面时已登录）后拉知识库信息，并把上一次的对话接回来。
watch(
  user,
  (current, previous) => {
    if (current) {
      void refresh()
      void restore()
      return
    }
    if (previous) {
      // 主动退出或登录过期：别把上一个账号的对话留在页面上。
      forget()
    }
  },
)

async function handleSend(query: string): Promise<void> {
  if (sending.value) {
    return
  }
  sidebarOpen.value = false
  await send(query, selectedLaw.value)
}

async function handleClear(): Promise<void> {
  if (!hasMessages.value) {
    return
  }

  try {
    await ElMessageBox.confirm(
      '清空后这条对话会从历史记录里一起删掉，确定吗？',
      '清空对话',
      {
        confirmButtonText: '清空',
        cancelButtonText: '再想想',
        type: 'warning',
      },
    )
  } catch {
    return
  }

  sidebarOpen.value = false
  try {
    await clearCurrent()
  } catch {
    ElMessage.error('这条对话没能删掉，请稍后再试。')
  }
}

function handleNewConversation(): void {
  startNew()
  sidebarOpen.value = false
}

async function handleSelectConversation(id: number): Promise<void> {
  try {
    await openConversation(id)
    sidebarOpen.value = false
  } catch (error) {
    ElMessage.error(
      error instanceof ApiError
        ? error.message
        : '这条对话打不开，请稍后重试。',
    )
  }
}

async function handleDeleteConversation(id: number): Promise<void> {
  try {
    await ElMessageBox.confirm(
      '删除后这条对话就找不回来了，确定吗？',
      '删除对话',
      {
        confirmButtonText: '删除',
        cancelButtonText: '再想想',
        type: 'warning',
      },
    )
  } catch {
    return
  }

  try {
    await removeConversation(id)
  } catch {
    ElMessage.error('这条对话没能删掉，请稍后再试。')
  }
}

function handleOpenUpload(): void {
  const missing = checks.value.filter((check) => !check.ok)
  if (missing.length > 0) {
    ElMessage.warning('索引还不完整，上传前请先完成一次重建。')
    return
  }
  uploadOpen.value = true
}

async function handleUploadCompleted(result: UploadResult): Promise<void> {
  ElMessage.success(`《${result.law_name}》已经加入知识库。`)
  await refresh()
}

async function handleRebuildCompleted(result: RebuildResult): Promise<void> {
  ElMessage.success(`知识库已更新：共 ${result.law_count} 部法律。`)
  await refresh()
}

async function handleLogout(): Promise<void> {
  try {
    await ElMessageBox.confirm(
      '退出后要重新输入账号密码才能进入，确定吗？',
      '退出登录',
      {
        confirmButtonText: '退出',
        cancelButtonText: '再想想',
        type: 'warning',
      },
    )
  } catch {
    return
  }

  sidebarOpen.value = false
  await signOut()
  forget()
}
</script>

<template>
  <div v-if="!authReady" class="app-boot">正在确认登录状态…</div>

  <LoginView v-else-if="!user" />

  <div v-else class="app-shell">
    <div
      v-if="sidebarOpen"
      class="sidebar-scrim"
      @click="sidebarOpen = false"
    />

    <AppSidebar
      :class="{ 'is-open': sidebarOpen }"
      :user="user"
      :laws="laws"
      :selected-law="selectedLaw"
      :health="health"
      :ready="ready"
      :checks="checks"
      :loading="loading"
      :has-messages="hasMessages"
      :conversations="conversations"
      :active-conversation-id="conversationId"
      :history-loading="loadingHistory"
      @update:selected-law="selectedLaw = $event"
      @clear="handleClear"
      @open-law-list="lawListOpen = true"
      @open-upload="handleOpenUpload"
      @open-rebuild="rebuildOpen = true"
      @new-conversation="handleNewConversation"
      @select-conversation="handleSelectConversation"
      @delete-conversation="handleDeleteConversation"
      @logout="handleLogout"
    />

    <main class="main">
      <header class="mobile-bar">
        <el-button text :icon="Expand" @click="sidebarOpen = true">
          知识库
        </el-button>
        <span class="mobile-brand">RAG Law</span>
        <span class="mobile-spacer" />
      </header>

      <div ref="scrollArea" class="main-scroll">
        <div class="main-inner">
          <el-alert
            v-if="kbError"
            type="error"
            :title="kbError"
            :closable="false"
            show-icon
            class="kb-error"
          />

          <HeroPanel
            v-if="!hasMessages"
            :law-count="laws.length"
            :disabled="sending"
            @send="handleSend"
          />
          <ChatThread v-else :messages="messages" />
        </div>
      </div>

      <div class="composer-dock">
        <div class="composer-inner">
          <ComposerBar :disabled="sending" @send="handleSend" />
        </div>
      </div>
    </main>

    <LawListDialog v-model="lawListOpen" :laws="laws" />
    <UploadLawDialog
      v-model="uploadOpen"
      @completed="handleUploadCompleted"
    />
    <RebuildDialog
      v-model="rebuildOpen"
      @completed="handleRebuildCompleted"
    />
  </div>
</template>
