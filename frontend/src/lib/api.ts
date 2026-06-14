/**
 * API client for LocalRAG backend.
 * All requests go through the Next.js rewrite proxy.
 */

const BASE = '/api';

export interface Session {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

export interface MessageSource {
  filename: string;
  chunk_index: number;
  page: number;
  score: number;
  snippet: string;
}

export interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  sources: MessageSource[] | null;
  created_at: string;
}

export interface SessionWithMessages extends Session {
  messages: Message[];
}

export interface Document {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  chunk_count: number;
  status: 'ready' | 'processing' | 'error';
  source: string;
  error_message: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface CollectionStats {
  collection_name: string;
  chunk_count: number;
  embed_model: string;
  persist_dir: string;
}

export interface HealthStatus {
  status: string;
  version: string;
  ollama_connected: boolean;
  chroma_chunks: number;
  sqlite_connected: boolean;
}

export interface ChatResponse {
  answer: string;
  sources: MessageSource[];
  session_id: string;
}

// -----------------------------------------------------------------------
// Generic fetch wrapper
// -----------------------------------------------------------------------

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {}
    throw new Error(`API error ${res.status}: ${detail}`);
  }

  if (res.status === 204) return undefined as T;
  return res.json();
}

// -----------------------------------------------------------------------
// Sessions
// -----------------------------------------------------------------------

export const sessionsApi = {
  list: () => apiFetch<Session[]>('/sessions'),

  create: (title?: string) =>
    apiFetch<Session>('/sessions', {
      method: 'POST',
      body: JSON.stringify({ title }),
    }),

  get: (id: string) => apiFetch<SessionWithMessages>(`/sessions/${id}`),

  rename: (id: string, title: string) =>
    apiFetch<Session>(`/sessions/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    }),

  delete: (id: string) =>
    apiFetch<void>(`/sessions/${id}`, { method: 'DELETE' }),
};

// -----------------------------------------------------------------------
// Chat
// -----------------------------------------------------------------------

export const chatApi = {
  send: (sessionId: string, message: string) =>
    apiFetch<ChatResponse>(`/sessions/${sessionId}/chat`, {
      method: 'POST',
      body: JSON.stringify({ message }),
    }),

  streamUrl: (sessionId: string, message: string) =>
    `/api/sessions/${sessionId}/chat/stream?message=${encodeURIComponent(message)}`,
};

// -----------------------------------------------------------------------
// Documents
// -----------------------------------------------------------------------

export const documentsApi = {
  list: (status = 'all') =>
    apiFetch<{ documents: Document[]; total: number }>(`/documents?status=${status}`),

  get: (id: string) => apiFetch<Document>(`/documents/${id}`),

  stats: () => apiFetch<CollectionStats>('/documents/stats'),

  delete: (id: string) =>
    apiFetch<void>(`/documents/${id}`, { method: 'DELETE' }),

  upload: async (file: File): Promise<Document> => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${BASE}/documents/upload`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail ?? `Upload failed: ${res.statusText}`);
    }
    return res.json();
  },
};

// -----------------------------------------------------------------------
// Health
// -----------------------------------------------------------------------

export const healthApi = {
  check: () => apiFetch<HealthStatus>('/health').catch(() => null),
};
