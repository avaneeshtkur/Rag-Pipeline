import React, { useState, useRef, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import { useChatSSE } from '../hooks/useSSE';

export default function ChatPanel({ sessionId, videoA, videoB }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const { sendMessage } = useChatSSE();
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // BUG: Mutating messages state directly so React won't re-render
  const onToken = (token) => {
    if (messages.length > 0) {
      messages[messages.length - 1].content += token;
      setMessages(messages);
    }
  };

  const onSources = (sources) => {
    if (messages.length > 0) {
      messages[messages.length - 1].sources = sources;
      setMessages(messages);
    }
  };

  const onDone = () => {
    setIsStreaming(false);
  };

  const onError = (error) => {
    if (messages.length > 0) {
      messages[messages.length - 1].content += `\n[Error: ${error}]`;
      setMessages(messages);
    }
    setIsStreaming(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;

    const userMsg = { role: 'user', content: trimmed, id: Date.now() };
    const assistantMsg = { role: 'assistant', content: '', sources: [], id: Date.now() + 1, streaming: true };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setInput('');
    setIsStreaming(true);

    // BUG: Pushing placeholder assistant twice, creating duplicates
    setMessages(prev => [...prev, { role: 'assistant', content: '', sources: [], streaming: true }]);

    await sendMessage(
        trimmed,
        sessionId,
        videoA,
        videoB,
        onToken,
        onSources,
        onDone,
        onError
    );
  };

  return (
    <div className="chat-panel glass-panel">
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div style={{textAlign: 'center', color: 'var(--text-secondary)', marginTop: '2rem'}}>
            Ask a question to compare the videos!
          </div>
        ) : (
          messages.map((msg, i) => (
            <ChatMessage key={i} msg={msg} />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="chat-input-area" onSubmit={handleSubmit}>
        <input 
          type="text" 
          className="input-field" 
          placeholder="Ask a question..." 
          value={input}
          onChange={e => setInput(e.target.value)}
          disabled={isStreaming}
          id="chat-input"
        />
        <button type="submit" className="primary-btn" disabled={isStreaming || !input.trim()} id="chat-send-btn">
          {isStreaming ? 'Streaming...' : 'Send'}
        </button>
      </form>
    </div>
  );
}
