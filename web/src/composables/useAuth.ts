import { computed, ref } from 'vue'

import {
  ApiError,
  fetchCurrentUser,
  login as loginRequest,
  logout as logoutRequest,
  setUnauthorizedHandler,
} from '@/api/client'
import type { CurrentUser, LoginRequest } from '@/types'

// 登录态全站只有一份，所以状态放在模块级，各组件共享同一组 ref。
const user = ref<CurrentUser | null>(null)
const ready = ref(false)
const submitting = ref(false)
const error = ref<string | null>(null)

let bootstrapPromise: Promise<void> | null = null
let handlerRegistered = false

function registerHandler(): void {
  if (handlerRegistered) {
    return
  }
  // 任意业务请求返回 401 都视为登录失效，直接退回登录页。
  setUnauthorizedHandler(() => {
    user.value = null
  })
  handlerRegistered = true
}

export function useAuth() {
  registerHandler()

  /** 首次进入页面时确认浏览器里是否已有有效登录态。 */
  async function bootstrap(): Promise<void> {
    if (bootstrapPromise) {
      return bootstrapPromise
    }

    ready.value = false
    const task = (async () => {
      try {
        user.value = await fetchCurrentUser()
      } catch {
        user.value = null
      } finally {
        ready.value = true
        bootstrapPromise = null
      }
    })()
    bootstrapPromise = task
    return task
  }

  async function signIn(credentials: LoginRequest): Promise<boolean> {
    if (submitting.value) {
      return false
    }

    submitting.value = true
    error.value = null
    try {
      user.value = await loginRequest(credentials)
      return true
    } catch (cause) {
      error.value =
        cause instanceof ApiError
          ? cause.message
          : '这次没能登录成功，请稍后再试。'
      return false
    } finally {
      submitting.value = false
    }
  }

  async function signOut(): Promise<void> {
    try {
      await logoutRequest()
    } catch {
      // 网络不通也要让界面回到未登录状态。
    } finally {
      user.value = null
    }
  }

  function clearError(): void {
    error.value = null
  }

  return {
    user,
    ready,
    submitting,
    error,
    signedIn: computed(() => user.value !== null),
    isAdmin: computed(() => user.value?.role === 'admin'),
    bootstrap,
    signIn,
    signOut,
    clearError,
  }
}
