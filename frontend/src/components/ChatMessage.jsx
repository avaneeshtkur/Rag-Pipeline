import React from 'react';

export default function ChatMessage({ msg }) {
  const isUser = msg.role === 'user';
  
  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-bubble">
        {msg.content}
      </div>
      {!isUser && msg.sources && msg.sources.length > 0 && (
        <div className="citations">
          {msg.sources.map((src, i) => (
            <div key={i} className="citation-pill" title={src.text_preview}>
              [Video {src.video_id}, Chunk {src.chunk_index}]
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
