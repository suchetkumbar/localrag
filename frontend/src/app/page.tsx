'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { PlusIcon, TrashIcon, FileTextIcon, SendIcon, PaperclipIcon, ChevronDownIcon, ServerIcon } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { sessionsApi, chatApi, documentsApi, healthApi } from '@/lib/api';
import type { Session, Message, MessageSource, Document, HealthStatus } from '@/lib/api';
import { formatDistanceToNow } from 'date-fns';

// ---------------------------------------------------------------------------
// Source citation badge
// ---------------------------------------------------------------------------

function SourceBadge({ source }: { source: MessageSource }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="inline-block mr-1 mb-1">
      <button
        onClick={() => setOpen(!open)}
        className="text-xs px-2 py-0.5 rounded-full bg-sky-900/60 text-sky-300 border border-sky-700/50 hover:bg-sky-800/60 transition-colors flex items-center gap-1"
      >
        <FileTextIcon size={10} />
        {source.filename}
        <span className="text-sky-500">p.{source.page}</span>
        <ChevronDownIcon size={10} className={open ? 'rotate-180' : ''} />
      </button>
      {open && (
        <div className="mt-1 p-2 bg-gray-800 border border-gray-700 rounded text-xs text-gray-300 max-w-sm">
          <div className="font-mono text-sky-400 mb-1">
            {source.filename} · chunk {source.chunk_index} · score {source.score.toFixed(3)}
          </div>
          <p className="text-gray-400 line-clamp-4">{source.snippet}</p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Message bubble
// ---------------------------------------------------------------------------

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[85%] ${isUser ? 'order-2' : ''}`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'bg-sky-600 text-white rounded-br-sm'
              : 'bg-gray-800 text-gray-100 rounded-bl-sm border border-gray-700'
          }`}
        >
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <div className="prose-invert-custom">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>
        {message.sources && message.sources.length > 0 && (
          <div className="mt-1.5 px-1">
            <span className="text-xs text-gray-500 mr-1">Sources:</span>
            {message.sources.map((s, i) => (
              <SourceBadge key={i} source={s} />
            ))}
          </div>
        )}
        <p className="text-xs text-gray-600 mt-1 px-1">
          {formatDistanceToNow(new Date(message.created_at), { addSuffix: true })}
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Typing indicator
// ---------------------------------------------------------------------------

function TypingIndicator({ streamText }: { streamText: string }) {
  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[85%] bg-gray-800 border border-gray-700 rounded-2xl rounded-bl-sm px-4 py-3">
        {streamText ? (
          <div className="prose-invert-custom text-sm">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamText}</ReactMarkdown>
            <span className="inline-block w-2 h-4 bg-sky-400 ml-0.5 animate-pulse" />
          </div>
        ) : (
          <div className="flex gap-1 items-center py-1">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="w-2 h-2 bg-sky-500 rounded-full animate-bounce"
                style={{ animationDelay: `${i * 0.15}s` }}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Document panel
// ---------------------------------------------------------------------------

function DocumentPanel({ onClose }: { onClose: () => void }) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await documentsApi.list();
      setDocs(res.documents);
    } catch (e: any) {
      setError(e.message);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      await documentsApi.upload(file);
      await refresh();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this document and all its chunks?')) return;
    try {
      await documentsApi.delete(id);
      await refresh();
    } catch (err: any) {
      setError(err.message);
    }
  };

  return (
    <div className="w-80 bg-gray-900 border-l border-gray-800 flex flex-col">
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <h2 className="font-semibold text-gray-100">Documents</h2>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-300 text-lg leading-none">×</button>
      </div>

      <div className="p-3 border-b border-gray-800">
        <label className={`flex items-center justify-center gap-2 w-full py-2 px-3 rounded-lg border-2 border-dashed cursor-pointer transition-colors text-sm ${uploading ? 'border-gray-700 text-gray-600' : 'border-sky-700 text-sky-400 hover:border-sky-500 hover:bg-sky-900/20'}`}>
          <PaperclipIcon size={14} />
          {uploading ? 'Uploading…' : 'Upload document'}
          <input ref={fileRef} type="file" className="hidden" accept=".pdf,.docx,.md,.txt" onChange={handleUpload} disabled={uploading} />
        </label>
        <p className="text-xs text-gray-600 text-center mt-1">PDF, DOCX, MD, TXT</p>
        {error && <p className="text-xs text-red-400 mt-1 text-center">{error}</p>}
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {docs.length === 0 && (
          <p className="text-center text-gray-600 text-sm mt-8">No documents yet.<br />Upload one or drop files<br />in the watch folder.</p>
        )}
        {docs.map((doc) => (
          <div key={doc.id} className="flex items-start gap-2 p-2 rounded-lg bg-gray-800/50 hover:bg-gray-800 group">
            <FileTextIcon size={14} className="text-sky-500 mt-0.5 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-xs text-gray-200 truncate">{doc.filename}</p>
              <p className="text-xs text-gray-500">
                {doc.chunk_count} chunks · {(doc.file_size / 1024).toFixed(1)}KB
              </p>
              <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                doc.status === 'ready' ? 'bg-green-900/50 text-green-400' :
                doc.status === 'error' ? 'bg-red-900/50 text-red-400' :
                'bg-yellow-900/50 text-yellow-400'
              }`}>
                {doc.status}
              </span>
            </div>
            <button
              onClick={() => handleDelete(doc.id)}
              className="opacity-0 group-hover:opacity-100 text-gray-600 hover:text-red-400 transition-all"
            >
              <TrashIcon size={12} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function HomePage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSession, setActiveSession] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamText, setStreamText] = useState('');
  const [showDocs, setShowDocs] = useState(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => bottomRef.current?.scrollIntoView({ behavior: 'smooth' });

  useEffect(() => { scrollToBottom(); }, [messages, streamText]);

  useEffect(() => {
    loadSessions();
    healthApi.check().then(setHealth);
  }, []);

  const loadSessions = async () => {
    const list = await sessionsApi.list().catch(() => []);
    setSessions(list);
  };

  const selectSession = async (id: string) => {
    setActiveSession(id);
    const s = await sessionsApi.get(id).catch(() => null);
    if (s) setMessages(s.messages);
  };

  const newSession = async () => {
    const s = await sessionsApi.create('New Chat');
    setSessions((prev) => [s, ...prev]);
    setActiveSession(s.id);
    setMessages([]);
  };

  const deleteSession = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await sessionsApi.delete(id).catch(() => {});
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSession === id) {
      setActiveSession(null);
      setMessages([]);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isStreaming) return;

    let sessionId = activeSession;
    if (!sessionId) {
      const s = await sessionsApi.create(input.slice(0, 60));
      setSessions((prev) => [s, ...prev]);
      sessionId = s.id;
      setActiveSession(s.id);
    }

    const userMsg: Message = {
      id: Date.now(),
      role: 'user',
      content: input,
      sources: null,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    const question = input;
    setInput('');
    setIsStreaming(true);
    setStreamText('');

    try {
      const url = chatApi.streamUrl(sessionId, question);
      const sse = new EventSource(url);
      let accumulated = '';
      let sources: MessageSource[] = [];

      sse.onmessage = (e) => {
        if (e.data === '[DONE]') {
          sse.close();
          setIsStreaming(false);
          const aiMsg: Message = {
            id: Date.now() + 1,
            role: 'assistant',
            content: accumulated,
            sources,
            created_at: new Date().toISOString(),
          };
          setMessages((prev) => [...prev, aiMsg]);
          setStreamText('');
          loadSessions();
          return;
        }
        try {
          const token = JSON.parse(e.data);
          accumulated += token;
          setStreamText(accumulated);
        } catch {}
      };

      sse.addEventListener('sources', (e: any) => {
        try { sources = JSON.parse(e.data); } catch {}
      });

      sse.onerror = () => {
        sse.close();
        setIsStreaming(false);
        setStreamText('');
      };
    } catch (err) {
      setIsStreaming(false);
      setStreamText('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="h-full flex">
      {/* Sidebar */}
      <div className="w-60 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0">
        <div className="p-3 border-b border-gray-800">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-7 h-7 rounded-lg bg-sky-600 flex items-center justify-center">
              <ServerIcon size={14} />
            </div>
            <span className="font-bold text-gray-100 text-sm">LocalRAG</span>
            <span className={`ml-auto w-2 h-2 rounded-full ${health?.ollama_connected ? 'bg-green-400' : 'bg-red-400'}`} title={health?.ollama_connected ? 'Ollama connected' : 'Ollama offline'} />
          </div>
          <button
            onClick={newSession}
            className="w-full flex items-center gap-2 px-3 py-2 bg-sky-600 hover:bg-sky-500 rounded-lg text-sm font-medium transition-colors"
          >
            <PlusIcon size={14} />
            New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
          {sessions.map((s) => (
            <div
              key={s.id}
              onClick={() => selectSession(s.id)}
              className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer text-sm transition-colors ${
                activeSession === s.id ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }`}
            >
              <span className="flex-1 truncate">{s.title || 'Chat'}</span>
              <button
                onClick={(e) => deleteSession(s.id, e)}
                className="opacity-0 group-hover:opacity-100 text-gray-600 hover:text-red-400"
              >
                <TrashIcon size={12} />
              </button>
            </div>
          ))}
          {sessions.length === 0 && (
            <p className="text-xs text-gray-600 text-center mt-4">No chats yet</p>
          )}
        </div>

        <div className="p-3 border-t border-gray-800">
          <button
            onClick={() => setShowDocs(!showDocs)}
            className="w-full flex items-center gap-2 px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm text-gray-300 transition-colors"
          >
            <FileTextIcon size={14} />
            Documents
            {health && (
              <span className="ml-auto text-xs text-gray-500">{health.chroma_chunks} chunks</span>
            )}
          </button>
        </div>
      </div>

      {/* Chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4">
          {messages.length === 0 && !isStreaming && (
            <div className="h-full flex flex-col items-center justify-center text-center">
              <div className="w-16 h-16 rounded-2xl bg-sky-600/20 border border-sky-600/30 flex items-center justify-center mb-4">
                <ServerIcon size={28} className="text-sky-400" />
              </div>
              <h1 className="text-xl font-semibold text-gray-200 mb-2">LocalRAG</h1>
              <p className="text-gray-500 text-sm max-w-sm">
                Ask questions about your documents. All processing is 100% local — your data never leaves your machine.
              </p>
              {health && !health.ollama_connected && (
                <div className="mt-4 px-4 py-3 bg-red-900/30 border border-red-700/50 rounded-lg text-sm text-red-400">
                  ⚠️ Ollama is not running. Start it with <code className="font-mono">ollama serve</code>
                </div>
              )}
            </div>
          )}
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          {isStreaming && <TypingIndicator streamText={streamText} />}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="border-t border-gray-800 p-4">
          <div className="flex gap-3 items-end">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents…"
              rows={1}
              className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm text-gray-100 placeholder-gray-500 resize-none focus:outline-none focus:border-sky-600 transition-colors"
              style={{ maxHeight: '120px', overflowY: 'auto' }}
              disabled={isStreaming}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || isStreaming}
              className="p-3 bg-sky-600 hover:bg-sky-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-xl transition-colors text-white"
            >
              <SendIcon size={16} />
            </button>
          </div>
          <p className="text-xs text-gray-600 mt-2 text-center">
            Enter to send · Shift+Enter for new line · Powered by Ollama locally
          </p>
        </div>
      </div>

      {/* Documents panel */}
      {showDocs && <DocumentPanel onClose={() => setShowDocs(false)} />}
    </div>
  );
}
