import React from 'react';

export default function VideoCard({ data, label }) {
  if (!data) return null;

  return (
    <div className="video-card glass-panel">
      <div className="video-header">
        <span className="video-title">{label} - {data.title}</span>
      </div>
      
      {label === "Video A" ? (
        <iframe 
          width="100%" 
          height="200" 
          src={`https://www.youtube.com/embed/${data.video_id}`} 
          title="YouTube video player" 
          frameBorder="0" 
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
          allowFullScreen
          style={{borderRadius: '8px', border: 'none'}}
        />
      ) : (
        <div style={{height: '200px', background: 'rgba(0,0,0,0.4)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
           <a href={data.url} target="_blank" rel="noreferrer" style={{color: '#a855f7', textDecoration: 'none', fontWeight: 'bold'}}>
             View Reel on Instagram
           </a>
        </div>
      )}

      <div className="video-meta">
        <div className="meta-stat">
          <div className="meta-label">Creator</div>
          <div className="meta-value">@{data.creator}</div>
        </div>
        <div className="meta-stat">
          <div className="meta-label">Engagement</div>
          <div className="meta-value">{(data.engagement_rate || 0).toFixed(2)}%</div>
        </div>
        <div className="meta-stat">
          <div className="meta-label">Views</div>
          <div className="meta-value">{(data.views || 0).toLocaleString()}</div>
        </div>
        <div className="meta-stat">
          <div className="meta-label">Likes</div>
          <div className="meta-value">{(data.likes || 0).toLocaleString()}</div>
        </div>
      </div>
    </div>
  );
}
