import React, { useState, useRef, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import { useChatSSE } from '../hooks/useSSE';

export default function ChatPanel({ sessionId, videoA, videoB }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const { sendMessage } = useChatSSE();
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // BUG 1 FIX (Cause D): Functional state update creating new objects so React re-renders
  const onToken = (token) => {
    setMessages(prev => {
      const updated = [...prev];
      const last = { ...updated[updated.length - 1] };
      last.content = (last.content || '') + token;
      updated[updated.length - 1] = last;
      return updated;
    });
  };

  const onSources = (sources) => {
    setMessages(prev => {
      const updated = [...prev];
      const last = { ...updated[updated.length - 1] };
      last.sources = sources;
      updated[updated.length - 1] = last;
      return updated;
    });
  };

  const onDone = () => {
    setIsStreaming(false);
  };

  const onError = (error) => {
    setMessages(prev => {
      const updated = [...prev];
      const last = { ...updated[updated.length - 1] };
      last.content = (last.content || '') + `\n[Error: ${error}]`;
      updated[updated.length - 1] = last;
      return updated;
    });
    setIsStreaming(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;

    const userMsg = { role: 'user', content: trimmed, id: Date.now() };
    const assistantMsg = { role: 'assistant', content: '', sources: [], id: Date.now() + 1, streaming: true };

    // Add user + empty assistant in ONE state update — never add assistant again after this
    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setInput('');
    setIsStreaming(true);

    await sendMessage(
        trimmed,
        sessionId,
        videoA,
        videoB,
        // onToken: append to the LAST message only — no new message is added
        (token) => {
            setMessages(prev => {
                const copy = [...prev];
                const last = { ...copy[copy.length - 1] };
                last.content = (last.content || '') + token;
                copy[copy.length - 1] = last;
                return copy;
            });
        },
        // onSources: attach to the LAST message only
        (sources) => {
            setMessages(prev => {
                const copy = [...prev];
                const last = { ...copy[copy.length - 1] };
                last.sources = sources;
                last.streaming = false;
                copy[copy.length - 1] = last;
                return copy;
            });
        },
        // onDone: mark last message as no longer streaming
        () => {
            setMessages(prev => {
                const copy = [...prev];
                const last = { ...copy[copy.length - 1] };
                last.streaming = false;
                copy[copy.length - 1] = last;
                return copy;
            });
            setIsStreaming(false);
        },
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
