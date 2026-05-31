import React, { useState, useEffect } from 'react';
import InputBar from './components/InputBar';
import VideoCard from './components/VideoCard';
import ChatPanel from './components/ChatPanel';

function App() {
  const [session, setSession] = useState({
    id: null,
    videoA: null,
    videoB: null
  });
  const [ollamaWarning, setOllamaWarning] = useState(false);

  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => {
        if (!data.ollama_running) {
          setOllamaWarning(true);
        }
      })
      .catch(() => setOllamaWarning(true));
  }, []);

  const handleAnalyze = (data) => {
    setSession({
      id: data.session_id,
      videoA: data.video_a,
      videoB: data.video_b
    });
  };

  return (
    <div className="app-container">
      {ollamaWarning && (
        <div className="ollama-warning" id="ollama-warning-banner">
          <span className="warning-icon">⚠️</span>
          <span>Ollama is not running. Open a terminal and run: <code>ollama serve</code></span>
          <button
            className="warning-dismiss"
            onClick={() => setOllamaWarning(false)}
            aria-label="Dismiss warning"
          >
            ✕
          </button>
        </div>
      )}

      <header className="header">
        <h1>RAG Video Analyzer</h1>
        <p>Compare engagement strategies of YouTube and Instagram Reels</p>
      </header>

      <InputBar onAnalyze={handleAnalyze} />

      {session.id && (
        <div className="main-content">
          <div className="videos-container">
            <VideoCard data={session.videoA} label="Video A" />
            <VideoCard data={session.videoB} label="Video B" />
          </div>
          
          <ChatPanel 
            sessionId={session.id} 
            videoA={session.videoA} 
            videoB={session.videoB} 
          />
        </div>
      )}
    </div>
  );
}

export default App;
