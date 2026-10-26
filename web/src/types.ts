/** 与后端 Pydantic 模型一一对应的前端类型。 */

export type AnswerStatus =
  | 'answered'
  | 'needs_clarification'
  | 'insufficient_evidence'
  | 'generation_failed'

export type RetrievalMethod = 'bm25' | 'dense' | 'hybrid' | 'rerank'

export type ContextRelation = 'matched' | 'adjacent'

export interface ConversationTurn {
  role: 'user' | 'assistant'
  content: string
}

export interface LawOption {
  law_id: string
  law_name: string
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  checks: Record<string, boolean>
}

export interface ValidatedCitation {
  evidence_id: string
  chunk_id: string
  law_id: string
  law_name: string
  article_no: string
  quote: string
  source_file: string
  page: number | null
  end_page: number | null
}

export interface AnswerResponse {
  status: AnswerStatus
  summary: string
  analysis: string
  conditions: string[]
  citations: ValidatedCitation[]
  limitations: string
  follow_up_question: string | null
}

export interface RetrievalResult {
  chunk_id: string
  law_id: string
  law_name: string
  article_no: string
  text: string
  part: string | null
  chapter: string | null
  section: string | null
  paragraph_no: number | null
  source_file: string
  page: number | null
  end_page: number | null
  score: number
  source: RetrievalMethod
  component_scores: Record<string, number>
}

export interface ContextEvidence {
  evidence_id: string
  chunk_id: string
  law_id: string
  law_name: string
  article_no: string
  text: string
  part: string | null
  chapter: string | null
  section: string | null
  source_file: string
  page: number | null
  end_page: number | null
  relation: ContextRelation
  is_excerpt: boolean
  score: number
}

export interface BuiltContext {
  text: string
  evidences: ContextEvidence[]
  estimated_tokens: number
  max_tokens: number
  truncated: boolean
}

export interface RetrievalContextResult {
  query: string
  reranked_results: RetrievalResult[]
  context: BuiltContext
  retrieval_latency_ms: number
  context_latency_ms: number
  total_latency_ms: number
}

export interface AnswerPipelineResult {
  query: string
  answer: AnswerResponse
  retrieval: RetrievalContextResult
  generation_attempts: number
  top_relevance_score: number | null
  validation_errors: string[]
  prompt_version: string
  generation_latency_ms: number
  total_latency_ms: number
}

export interface QueryRequest {
  query: string
  law_name: string | null
  history: ConversationTurn[]
}

export interface UploadResult {
  law_id: string
  law_name: string
  article_count: number
  chunk_count: number
  law_count: number
  chunk_total: number
  manifest_updated: boolean
}

export interface RebuildResult {
  law_count: number
  chunk_total: number
}

export interface ProgressEvent {
  stage: string
  message: string
}

export interface StreamErrorPayload {
  message: string
  detail?: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content?: string
  result?: AnswerPipelineResult
  error?: boolean
  pending?: boolean
  lawName?: string | null
}
