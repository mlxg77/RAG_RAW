<script setup lang="ts">
import {
  Document as DocumentIcon,
  UploadFilled,
} from '@element-plus/icons-vue'
import {
  ElMessage,
  genFileId,
  type UploadFile,
  type UploadInstance,
  type UploadRawFile,
} from 'element-plus'
import { computed, ref, watch } from 'vue'

import { uploadLaw } from '@/api/stream'
import type { ProgressEvent, UploadResult } from '@/types'
import ProgressSteps from './ProgressSteps.vue'

const MAX_BYTES = 50 * 1024 * 1024
const LAW_ID_PATTERN = /^cn_[a-z0-9_]+$/

const visible = defineModel<boolean>({ default: false })
const emit = defineEmits<{ completed: [result: UploadResult] }>()

const uploadRef = ref<UploadInstance>()
const selectedFile = ref<File | null>(null)
const lawId = ref('')
const lawName = ref('')
const events = ref<ProgressEvent[]>([])
const state = ref<'idle' | 'running' | 'complete' | 'error'>('idle')
const result = ref<UploadResult | null>(null)
const errorMessage = ref('')
const errorDetail = ref('')

const busy = computed(() => state.value === 'running')
const lawIdError = computed(() => {
  const value = lawId.value.trim()
  if (!value || LAW_ID_PATTERN.test(value)) {
    return ''
  }
  return '要用 cn_ 开头，后面只能是小写字母、数字或下划线。'
})
const canSubmit = computed(
  () =>
    selectedFile.value !== null &&
    LAW_ID_PATTERN.test(lawId.value.trim()) &&
    !busy.value,
)
const successMessage = computed(() => {
  if (!result.value) {
    return ''
  }
  return (
    `已收录《${result.value.law_name}》：` +
    `${result.value.article_count} 条法条，` +
    `知识库共 ${result.value.law_count} 部法律、` +
    `${result.value.chunk_total} 个 chunk。`
  )
})

watch(visible, (open) => {
  if (open) {
    reset()
  }
})

function reset(): void {
  uploadRef.value?.clearFiles()
  selectedFile.value = null
  lawId.value = ''
  lawName.value = ''
  events.value = []
  result.value = null
  errorMessage.value = ''
  errorDetail.value = ''
  state.value = 'idle'
}

function formatSize(size: number): string {
  if (size < 1024 * 1024) {
    return `${Math.max(1, Math.round(size / 1024))} KB`
  }
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}

function acceptFile(file: File): boolean {
  if (file.size > MAX_BYTES) {
    ElMessage.warning('文件太大了，请控制在 50 MB 以内。')
    uploadRef.value?.clearFiles()
    selectedFile.value = null
    return false
  }

  selectedFile.value = file
  if (!lawName.value.trim()) {
    lawName.value = file.name.replace(/\.[^.]+$/, '')
  }
  return true
}

function handleChange(uploadFile: UploadFile): void {
  const raw = uploadFile.raw
  if (raw) {
    acceptFile(raw)
  }
}

function handleExceed(files: File[]): void {
  uploadRef.value?.clearFiles()
  const file = files[0] as UploadRawFile
  file.uid = genFileId()
  uploadRef.value?.handleStart(file)
}

function clearFile(): void {
  uploadRef.value?.clearFiles()
  selectedFile.value = null
}

function handleBeforeClose(done: () => void): void {
  if (!busy.value) {
    done()
  }
}

async function submit(): Promise<void> {
  const file = selectedFile.value
  if (!file || !canSubmit.value) {
    return
  }

  events.value = []
  result.value = null
  errorMessage.value = ''
  errorDetail.value = ''
  state.value = 'running'

  await uploadLaw(file, lawId.value.trim(), lawName.value.trim(), {
    onProgress: (event) => events.value.push(event),
    onComplete: (payload) => {
      result.value = payload
      state.value = 'complete'
      emit('completed', payload)
    },
    onError: (payload) => {
      state.value = 'error'
      errorMessage.value = payload.message
      errorDetail.value = payload.detail ?? ''
    },
  })
}
</script>

<template>
  <el-dialog
    v-model="visible"
    title="添加新法律"
    width="600px"
    :close-on-click-modal="false"
    :close-on-press-escape="!busy"
    :before-close="handleBeforeClose"
  >
    <p class="dialog-lead">
      上传文件后会自动清洗、切分并重建 BM25 与向量索引，
      过程中会调用 Embedding 服务，可能需要几分钟。
    </p>

    <el-form label-position="top" class="dialog-form">
      <el-form-item label="法律文件" required>
        <div class="upload-area">
          <el-upload
            ref="uploadRef"
            drag
            action="#"
            :auto-upload="false"
            :limit="1"
            :show-file-list="false"
            accept=".pdf,.docx,.md"
            :on-change="handleChange"
            :on-exceed="handleExceed"
            :disabled="busy"
          >
            <el-icon class="upload-icon"><UploadFilled /></el-icon>
            <div class="el-upload__text">
              把法律文件拖到这里，或 <em>点击选择</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                支持 PDF、DOCX 和 Markdown，单个文件不超过 50 MB。
              </div>
            </template>
          </el-upload>

          <div v-if="selectedFile" class="selected-file">
            <el-icon><DocumentIcon /></el-icon>
            <span class="selected-file-name">{{ selectedFile.name }}</span>
            <span class="selected-file-size">
              {{ formatSize(selectedFile.size) }}
            </span>
            <el-button
              v-if="!busy"
              link
              type="danger"
              @click="clearFile"
            >
              移除
            </el-button>
          </div>
        </div>
      </el-form-item>

      <el-form-item
        label="law_id"
        required
        :error="lawIdError || undefined"
      >
        <el-input
          v-model="lawId"
          placeholder="cn_company_law"
          :disabled="busy"
        />
      </el-form-item>

      <el-form-item label="法律名称（可选）">
        <el-input
          v-model="lawName"
          placeholder="中华人民共和国公司法"
          :disabled="busy"
        />
      </el-form-item>
    </el-form>

    <ProgressSteps
      v-if="state !== 'idle'"
      :events="events"
      :state="state"
    />

    <el-alert
      v-if="successMessage"
      type="success"
      :title="successMessage"
      :closable="false"
      show-icon
      class="dialog-alert"
    />

    <el-alert
      v-if="errorMessage"
      type="error"
      :title="errorMessage"
      :closable="false"
      show-icon
      class="dialog-alert"
    />

    <el-collapse v-if="errorDetail" class="detail-collapse">
      <el-collapse-item title="技术信息" name="detail">
        <p class="detail-text">{{ errorDetail }}</p>
      </el-collapse-item>
    </el-collapse>

    <template #footer>
      <el-button :disabled="busy" @click="visible = false">取消</el-button>
      <el-button
        type="primary"
        :loading="busy"
        :disabled="!canSubmit || state === 'complete'"
        @click="submit"
      >
        {{ state === 'complete' ? '已完成' : '上传并重建索引' }}
      </el-button>
    </template>
  </el-dialog>
</template>
