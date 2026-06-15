'use client';

import { formatDistanceToNow } from 'date-fns';
import {
  AlertTriangleIcon,
  CheckCircle2Icon,
  ChevronDownIcon,
  CircleIcon,
  DatabaseIcon,
  FileTextIcon,
  Loader2Icon,
  MessageSquareIcon,
  PanelRightIcon,
  PaperclipIcon,
  PlusIcon,
  SendIcon,
  ServerIcon,
  TrashIcon,
  UploadCloudIcon,
  WifiIcon,
  WifiOffIcon,
  XIcon,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { chatApi, documentsApi, healthApi, sessionsApi } from '@/lib/api';
import type { Document, HealthStatus, Message, MessageSource, Session } from '@/lib/api';

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return 'Something went wrong';
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function relativeDate(value: string | null): string {
  if (!value) return 'Unknown';
  return formatDistanceToNow(new Date(value), { addSuffix: true });
}

function SourceBadge({ source }: { source: MessageSource }) {
  const [open, setOpen] = useState(false);

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="inline-flex h-7 items-center gap-1 rounded-md border border-cyan-400/20 bg-cyan-400/10 px-2 text-xs font-medium text-cyan-100 transition hover:border-cyan-300/50 hover:bg-cyan-400/15"
        title="View source"
      >
        <FileTextIcon size={12} />
        <span className="max-w-32 truncate">{source.filename}</span>
        <span className="text-cyan-300/70">p.{source.page}</span>
        <ChevronDownIcon size={12} className={open ? 'rotate-180 transition' : 'transition'} />
      </button>
      {open && (
        <span className="absolute left-0 top-9 z-20 block w-80 rounded-lg border border-zinc-700 bg-zinc-950 p-3 text-xs text-zinc-300 shadow-2xl shadow-black/40">
          <span className="mb-2 block font-mono text-cyan-300">
            chunk {source.chunk_index} / score {source.score.toFixed(3)}
          </span>
          <span className="line-clamp-5 text-zinc-400">{source.snippet}</span>
        </span>
      )}
    </span>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';

  return (
    <article className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[min(760px,86%)] ${isUser ? 'items-end' : 'items-start'} flex flex-col gap-2`}>
        <div
          className={
            isUser
              ? 'rounded-lg bg-cyan-500 px-4 py-3 text-sm leading-relaxed text-zinc-950 shadow-lg shadow-cyan-950/20'
              : 'rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 text-sm leading-relaxed text-zinc-100 shadow-lg shadow-black/20'
          }
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose-invert-custom">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {message.sources && message.sources.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {message.sources.map((source) => (
              <SourceBadge
                key={`${source.filename}-${source.chunk_index}-${source.page}`}
                source={source}
              />
            ))}
          </div>
        )}

        <span className="px-1 text-xs text-zinc-600">{relativeDate(message.created_at)}</span>
      </div>
    </article>
  );
}

function TypingIndicator({ streamText }: { streamText: string }) {
  return (
    <article className="flex justify-start">
      <div className="max-w-[min(760px,86%)] rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 text-sm text-zinc-100 shadow-lg shadow-black/20">
        {streamText ? (
          <div className="prose-invert-custom">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamText}</ReactMarkdown>
            <span className="ml-1 inline-block h-4 w-1.5 translate-y-0.5 animate-pulse rounded-full bg-cyan-300" />
          </div>
        ) : (
          <div className="flex items-center gap-2 text-zinc-400">
            <Loader2Icon size={14} className="animate-spin text-cyan-300" />
            <span>Thinking</span>
          </div>
        )}
      </div>
    </article>
  );
}

function StatusPill({ health }: { health: HealthStatus | null }) {
  const connected = Boolean(health?.ollama_connected);

  return (
    <div
      className={`inline-flex h-8 items-center gap-2 rounded-md border px-2.5 text-xs ${
        connected
          ? 'border-emerald-400/20 bg-emerald-400/10 text-emerald-200'
          : 'border-red-400/20 bg-red-400/10 text-red-200'
      }`}
    >
      {connected ? <WifiIcon size={13} /> : <WifiOffIcon size={13} />}
      <span>{connected ? 'Ollama online' : 'Ollama offline'}</span>
    </div>
  );
}

function DocumentStatus({ status }: { status: Document['status'] }) {
  const styles = {
    ready: 'border-emerald-400/20 bg-emerald-400/10 text-emerald-200',
    processing: 'border-amber-400/20 bg-amber-400/10 text-amber-200',
    error: 'border-red-400/20 bg-red-400/10 text-red-200',
  };

  const Icon = status === 'ready' ? CheckCircle2Icon : status === 'error' ? AlertTriangleIcon : Loader2Icon;

  return (
    <span className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs ${styles[status]}`}>
      <Icon size={12} className={status === 'processing' ? 'animate-spin' : ''} />
      {status}
    </span>
  );
}

function DocumentPanel({
  onClose,
  onDocumentsChanged,
  health,
}: {
  onClose: () => void;
  onDocumentsChanged: () => void;
  health: HealthStatus | null;
}) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const canUpload = health?.ollama_connected ?? true;

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const result = await documentsApi.list();
      setDocs(result.documents);
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!canUpload) {
      setError('Ollama is offline. Start Ollama before uploading documents.');
      if (fileRef.current) fileRef.current.value = '';
      return;
    }

    setUploading(true);
    setError(null);
    try {
      await documentsApi.upload(file);
      await refresh();
      onDocumentsChanged();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this document and its chunks?')) return;

    try {
      await documentsApi.delete(id);
      await refresh();
      onDocumentsChanged();
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  return (
    <aside className="w-full border-l border-zinc-800 bg-zinc-950/95 md:w-[380px]">
      <div className="flex h-full flex-col">
        <header className="flex h-16 items-center justify-between border-b border-zinc-800 px-5">
          <div>
            <h2 className="text-sm font-semibold text-zinc-100">Documents</h2>
            <p className="text-xs text-zinc-500">{docs.length} indexed</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="grid h-8 w-8 place-items-center rounded-md text-zinc-500 transition hover:bg-zinc-900 hover:text-zinc-100"
            title="Close documents"
          >
            <XIcon size={16} />
          </button>
        </header>

        <div className="border-b border-zinc-800 p-4">
          <label
            className={`group flex min-h-28 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-4 text-center transition ${
            uploading
              ? 'border-zinc-700 bg-zinc-900/60 text-zinc-500'
              : !canUpload
                ? 'cursor-not-allowed border-red-400/20 bg-red-400/5 text-red-100'
              : 'border-cyan-400/40 bg-cyan-400/[0.04] text-cyan-100 hover:border-cyan-300 hover:bg-cyan-400/[0.08]'
            }`}
          >
            {uploading ? (
              <Loader2Icon size={22} className="mb-2 animate-spin text-cyan-300" />
            ) : (
              <UploadCloudIcon size={24} className="mb-2 text-cyan-300" />
            )}
            <span className="text-sm font-medium">
              {uploading ? 'Uploading' : canUpload ? 'Upload document' : 'Ollama offline'}
            </span>
            <span className="mt-1 text-xs text-zinc-500">
              {canUpload ? 'PDF, DOCX, MD, TXT' : 'Embeddings are unavailable'}
            </span>
            <input
              ref={fileRef}
              type="file"
              className="hidden"
              accept=".pdf,.docx,.md,.txt"
              onChange={handleUpload}
              disabled={uploading || !canUpload}
            />
          </label>

          {error && (
            <div className="mt-3 rounded-lg border border-red-400/20 bg-red-400/10 p-3 text-xs leading-relaxed text-red-100">
              {error}
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-3">
          {loading && (
            <div className="flex items-center justify-center gap-2 py-10 text-sm text-zinc-500">
              <Loader2Icon size={16} className="animate-spin" />
              Loading documents
            </div>
          )}

          {!loading && docs.length === 0 && (
            <div className="mt-10 flex flex-col items-center text-center">
              <div className="grid h-12 w-12 place-items-center rounded-lg border border-zinc-800 bg-zinc-900 text-zinc-500">
                <FileTextIcon size={20} />
              </div>
              <p className="mt-4 text-sm font-medium text-zinc-300">No documents indexed</p>
              <p className="mt-1 max-w-52 text-xs leading-relaxed text-zinc-600">The watch folder and uploads will appear here.</p>
            </div>
          )}

          <div className="space-y-2">
            {docs.map((doc) => (
              <div
                key={doc.id}
                className="group rounded-lg border border-zinc-800 bg-zinc-900/70 p-3 transition hover:border-zinc-700 hover:bg-zinc-900"
              >
                <div className="flex items-start gap-3">
                  <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-zinc-950 text-cyan-300">
                    <FileTextIcon size={16} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-zinc-100">{doc.filename}</p>
                    <p className="mt-0.5 text-xs text-zinc-500">
                      {formatBytes(doc.file_size)} / {doc.chunk_count} chunks
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleDelete(doc.id)}
                    className="grid h-8 w-8 place-items-center rounded-md text-zinc-600 opacity-0 transition hover:bg-red-400/10 hover:text-red-300 group-hover:opacity-100"
                    title="Delete document"
                  >
                    <TrashIcon size={14} />
                  </button>
                </div>
                <div className="mt-3 flex items-center justify-between gap-2">
                  <DocumentStatus status={doc.status} />
                  <span className="truncate text-xs text-zinc-600">{relativeDate(doc.created_at)}</span>
                </div>
                {doc.error_message && (
                  <p className="mt-2 rounded-md bg-red-400/10 px-2 py-1 text-xs text-red-200">
                    {doc.error_message}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </aside>
  );
}

export default function HomePage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSession, setActiveSession] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamText, setStreamText] = useState('');
  const [showDocs, setShowDocs] = useState(true);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [appError, setAppError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const activeSessionTitle = useMemo(() => {
    return sessions.find((session) => session.id === activeSession)?.title ?? 'New chat';
  }, [activeSession, sessions]);

  const refreshHealth = useCallback(async () => {
    setHealth(await healthApi.check());
  }, []);

  const loadSessions = useCallback(async () => {
    try {
      const list = await sessionsApi.list();
      setSessions(list);
      setAppError(null);
    } catch (err) {
      setAppError(getErrorMessage(err));
      setSessions([]);
    }
  }, []);

  useEffect(() => {
    loadSessions();
    refreshHealth();
  }, [loadSessions, refreshHealth]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamText]);

  const selectSession = async (id: string) => {
    setActiveSession(id);
    setAppError(null);
    try {
      const session = await sessionsApi.get(id);
      setMessages(session.messages);
    } catch (err) {
      setAppError(getErrorMessage(err));
    }
  };

  const newSession = async () => {
    setAppError(null);
    try {
      const session = await sessionsApi.create('New Chat');
      setSessions((prev) => [session, ...prev]);
      setActiveSession(session.id);
      setMessages([]);
    } catch (err) {
      setAppError(getErrorMessage(err));
    }
  };

  const deleteSession = async (id: string, event: React.MouseEvent) => {
    event.stopPropagation();
    try {
      await sessionsApi.delete(id);
      setSessions((prev) => prev.filter((session) => session.id !== id));
      if (activeSession === id) {
        setActiveSession(null);
        setMessages([]);
      }
    } catch (err) {
      setAppError(getErrorMessage(err));
    }
  };

  const sendMessage = async () => {
    const question = input.trim();
    if (!question || isStreaming) return;

    setAppError(null);
    let sessionId = activeSession;

    try {
      if (!sessionId) {
        const session = await sessionsApi.create(question.slice(0, 60));
        setSessions((prev) => [session, ...prev]);
        sessionId = session.id;
        setActiveSession(session.id);
      }

      const userMessage: Message = {
        id: Date.now(),
        role: 'user',
        content: question,
        sources: null,
        created_at: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setInput('');
      setIsStreaming(true);
      setStreamText('');

      const sse = new EventSource(chatApi.streamUrl(sessionId, question));
      let accumulated = '';
      let sources: MessageSource[] = [];

      sse.onmessage = (event) => {
        if (event.data === '[DONE]') {
          sse.close();
          setIsStreaming(false);
          setMessages((prev) => [
            ...prev,
            {
              id: Date.now() + 1,
              role: 'assistant',
              content: accumulated,
              sources,
              created_at: new Date().toISOString(),
            },
          ]);
          setStreamText('');
          loadSessions();
          return;
        }

        try {
          const token = JSON.parse(event.data) as string;
          accumulated += token;
          setStreamText(accumulated);
        } catch {
          setAppError('The stream returned an invalid response.');
        }
      };

      sse.addEventListener('sources', (event: MessageEvent) => {
        try {
          sources = JSON.parse(event.data) as MessageSource[];
        } catch {
          sources = [];
        }
      });

      sse.onerror = () => {
        sse.close();
        setIsStreaming(false);
        setStreamText('');
        setAppError('The response stream stopped unexpectedly.');
      };
    } catch (err) {
      setIsStreaming(false);
      setStreamText('');
      setAppError(getErrorMessage(err));
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <main className="flex h-full bg-zinc-950 text-zinc-100">
      <aside className="flex w-[280px] shrink-0 flex-col border-r border-zinc-800 bg-[#10100f]">
        <div className="border-b border-zinc-800 p-4">
          <div className="mb-4 flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-lg bg-cyan-400 text-zinc-950">
              <ServerIcon size={18} />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-white">LocalRAG</p>
              <p className="text-xs text-zinc-500">{health?.chroma_chunks ?? 0} chunks indexed</p>
            </div>
            <span
              className={`h-2.5 w-2.5 rounded-full ${health?.ollama_connected ? 'bg-emerald-400' : 'bg-red-400'}`}
              title={health?.ollama_connected ? 'Ollama online' : 'Ollama offline'}
            />
          </div>

          <button
            type="button"
            onClick={newSession}
            className="flex h-10 w-full items-center justify-center gap-2 rounded-lg bg-cyan-400 text-sm font-semibold text-zinc-950 transition hover:bg-cyan-300"
          >
            <PlusIcon size={16} />
            New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3">
          <div className="mb-2 flex items-center justify-between px-1">
            <span className="text-xs font-medium uppercase tracking-wide text-zinc-600">Chats</span>
            <span className="text-xs text-zinc-600">{sessions.length}</span>
          </div>

          <div className="space-y-1">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={`group flex w-full items-center gap-1 rounded-lg px-2 py-1 transition ${
                  activeSession === session.id
                    ? 'bg-zinc-800'
                    : 'text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100'
                }`}
              >
                <button
                  type="button"
                  onClick={() => selectSession(session.id)}
                  className={`flex min-w-0 flex-1 items-center gap-2 rounded-md px-1 py-1 text-left text-sm ${
                    activeSession === session.id ? 'text-white' : 'text-inherit'
                  }`}
                >
                  <MessageSquareIcon size={15} className="shrink-0 text-zinc-500" />
                  <span className="min-w-0 flex-1 truncate">{session.title || 'Untitled chat'}</span>
                </button>
                <button
                  type="button"
                  onClick={(event) => deleteSession(session.id, event)}
                  className="grid h-7 w-7 place-items-center rounded-md text-zinc-600 opacity-0 transition hover:bg-red-400/10 hover:text-red-300 group-hover:opacity-100"
                  title="Delete chat"
                >
                  <TrashIcon size={13} />
                </button>
              </div>
            ))}
          </div>

          {sessions.length === 0 && (
            <div className="mt-8 rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 text-center">
              <MessageSquareIcon size={18} className="mx-auto text-zinc-600" />
              <p className="mt-2 text-sm text-zinc-400">No conversations</p>
            </div>
          )}
        </div>

        <div className="border-t border-zinc-800 p-3">
          <button
            type="button"
            onClick={() => setShowDocs((value) => !value)}
            className={`flex h-11 w-full items-center gap-2 rounded-lg px-3 text-sm transition ${
              showDocs ? 'bg-zinc-800 text-white' : 'bg-zinc-900 text-zinc-300 hover:bg-zinc-800'
            }`}
          >
            <DatabaseIcon size={16} />
            Documents
            <PanelRightIcon size={15} className="ml-auto text-zinc-500" />
          </button>
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col bg-[#080908]">
        <header className="flex h-16 items-center justify-between border-b border-zinc-800 px-5">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <CircleIcon size={8} className="fill-cyan-300 text-cyan-300" />
              <h1 className="truncate text-sm font-semibold text-zinc-100">{activeSessionTitle}</h1>
            </div>
            <p className="mt-1 text-xs text-zinc-600">{messages.length} messages</p>
          </div>
          <StatusPill health={health} />
        </header>

        {appError && (
          <div className="mx-5 mt-4 rounded-lg border border-red-400/20 bg-red-400/10 px-4 py-3 text-sm text-red-100">
            {appError}
          </div>
        )}

        <div className="flex-1 overflow-y-auto px-5 py-6">
          {messages.length === 0 && !isStreaming ? (
            <div className="mx-auto flex h-full max-w-3xl flex-col justify-center">
              <div className="mb-5 grid h-14 w-14 place-items-center rounded-lg border border-zinc-800 bg-zinc-900 text-cyan-300">
                <ServerIcon size={24} />
              </div>
              <h2 className="text-3xl font-semibold tracking-tight text-white">Ask your local archive.</h2>
              <div className="mt-6 grid gap-3 sm:grid-cols-3">
                {['Summarize the newest document', 'Find citations about a topic', 'Compare two uploaded files'].map(
                  (prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      onClick={() => setInput(prompt)}
                      className="rounded-lg border border-zinc-800 bg-zinc-900/70 p-3 text-left text-sm text-zinc-300 transition hover:border-cyan-400/40 hover:text-white"
                    >
                      {prompt}
                    </button>
                  ),
                )}
              </div>
              {health && !health.ollama_connected && (
                <div className="mt-6 flex items-center gap-2 rounded-lg border border-red-400/20 bg-red-400/10 px-4 py-3 text-sm text-red-100">
                  <AlertTriangleIcon size={16} />
                  Start Ollama before asking questions.
                </div>
              )}
            </div>
          ) : (
            <div className="mx-auto flex max-w-4xl flex-col gap-5">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              {isStreaming && <TypingIndicator streamText={streamText} />}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="border-t border-zinc-800 bg-[#10100f] px-5 py-4">
          <div className="mx-auto flex max-w-4xl items-end gap-3">
            <div className="flex-1 rounded-lg border border-zinc-700 bg-zinc-900 transition focus-within:border-cyan-400/70">
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question..."
                rows={1}
                className="max-h-32 min-h-12 w-full resize-none bg-transparent px-4 py-3 text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
                disabled={isStreaming}
              />
            </div>
            <button
              type="button"
              onClick={sendMessage}
              disabled={!input.trim() || isStreaming}
              className="grid h-12 w-12 shrink-0 place-items-center rounded-lg bg-cyan-400 text-zinc-950 transition hover:bg-cyan-300 disabled:bg-zinc-800 disabled:text-zinc-600"
              title="Send"
            >
              {isStreaming ? <Loader2Icon size={18} className="animate-spin" /> : <SendIcon size={18} />}
            </button>
          </div>
        </div>
      </section>

      {showDocs && (
        <DocumentPanel
          health={health}
          onClose={() => setShowDocs(false)}
          onDocumentsChanged={refreshHealth}
        />
      )}
    </main>
  );
}
