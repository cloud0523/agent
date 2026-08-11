import { useState, useRef, useEffect } from 'react';
import { apiQueryStream } from '../api/client';

export default function ChatPanel() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  function handleSend() {
    const question = input.trim();
    if (!question || streaming) return;

    const userMsg = { role: 'user', content: question };
    const assistantMsg = { role: 'assistant', content: '', sources: [] };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput('');
    setStreaming(true);

    apiQueryStream(
      { question, stream: true },
      (data) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = { ...updated[updated.length - 1] };
          if (data.type === 'token') {
            last.content += data.data || '';
          } else if (data.type === 'citation') {
            last.sources = data.sources || last.sources;
          }
          updated[updated.length - 1] = last;
          return updated;
        });
      },
      () => setStreaming(false),
      (err) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = { ...updated[updated.length - 1] };
          last.content = `❌ 错误: ${err.message}`;
          updated[updated.length - 1] = last;
          return updated;
        });
        setStreaming(false);
      }
    );
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">👋 有什么可以帮你的？</div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message message--${msg.role}`}>
            <div className="message__content">{msg.content}</div>
            {msg.sources?.length > 0 && (
              <div className="message__sources">
                📚 来源: {msg.sources.map((s) => s.filename).join(', ')}
              </div>
            )}
          </div>
        ))}
        {streaming && <div className="streaming-indicator">● 正在生成...</div>}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入你的问题... (Enter 发送, Shift+Enter 换行)"
          rows={2}
          disabled={streaming}
        />
        <button onClick={handleSend} disabled={streaming || !input.trim()}>
          发送
        </button>
      </div>
    </div>
  );
}
