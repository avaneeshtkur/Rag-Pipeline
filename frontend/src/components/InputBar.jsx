import React, { useState } from 'react';

export default function InputBar({ onAnalyze }) {
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [instagramUrl, setInstagramUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!youtubeUrl || !instagramUrl) return;
    
    setLoading(true);
    setError('');
    
    try {
      const res = await fetch('/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ youtube_url: youtubeUrl, instagram_url: instagramUrl })
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to ingest videos');
      
      onAnalyze(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="input-section glass-panel">
      {loading && (
        <div className="loading-overlay">
          <div className="loader"></div>
          <p>Analyzing Transcripts & Metadata... (this can take 30-60s)</p>
        </div>
      )}
      <form className="input-group" onSubmit={handleSubmit}>
        <input 
          type="url" 
          className="input-field" 
          placeholder="YouTube Video URL" 
          value={youtubeUrl} 
          onChange={e => setYoutubeUrl(e.target.value)} 
          required 
        />
        <input 
          type="url" 
          className="input-field" 
          placeholder="Instagram Reel URL" 
          value={instagramUrl} 
          onChange={e => setInstagramUrl(e.target.value)} 
          required 
        />
        <button type="submit" className="primary-btn" disabled={loading}>
          Analyze Videos
        </button>
      </form>
      {error && <div className="error-toast">{error}</div>}
    </div>
  );
}
