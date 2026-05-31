import React from 'react';

function formatDuration(seconds) {
  if (!seconds) return 'N/A';
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function formatDate(dateStr) {
  if (!dateStr) return 'N/A';
  // yt-dlp returns YYYYMMDD
  if (/^\d{8}$/.test(dateStr)) {
    return `${dateStr.slice(0,4)}-${dateStr.slice(4,6)}-${dateStr.slice(6,8)}`;
  }
  return dateStr;
}

export default function VideoCard({ data, label }) {
  if (!data) return null;

  const hashtags = Array.isArray(data.hashtags)
    ? data.hashtags
    : typeof data.hashtags === 'string' && data.hashtags
      ? data.hashtags.split(',').map(t => t.trim()).filter(Boolean)
      : [];

  return (
    <div className="video-card glass-panel">
      <div className="video-header">
        <span className="video-title">{label} - {data.title}</span>
      </div>
      
      {/* Display a simple iframe for YouTube, or link for Instagram */}
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
          <div className="meta-value" style={{fontSize: '1rem'}}>@{data.creator}</div>
        </div>
        <div className="meta-stat">
          <div className="meta-label">Followers</div>
          <div className="meta-value">{(data.followers || 0).toLocaleString()}</div>
        </div>
        <div className="meta-stat">
          <div className="meta-label">Views</div>
          <div className="meta-value">{(data.views || 0).toLocaleString()}</div>
        </div>
        <div className="meta-stat">
          <div className="meta-label">Likes</div>
          <div className="meta-value">{(data.likes || 0).toLocaleString()}</div>
        </div>
        <div className="meta-stat">
          <div className="meta-label">Comments</div>
          <div className="meta-value">{(data.comments || 0).toLocaleString()}</div>
        </div>
        <div className="meta-stat">
          <div className="meta-label">Engagement</div>
          <div className="meta-value">{(data.engagement_rate || 0).toFixed(2)}%</div>
        </div>
        <div className="meta-stat">
          <div className="meta-label">Duration</div>
          <div className="meta-value">{formatDuration(data.duration)}</div>
        </div>
        <div className="meta-stat">
          <div className="meta-label">Uploaded</div>
          <div className="meta-value" style={{fontSize: '0.9rem'}}>{formatDate(data.upload_date)}</div>
        </div>
      </div>

      {hashtags.length > 0 && (
        <div className="video-hashtags">
          {hashtags.slice(0, 8).map((tag, i) => (
            <span key={i} className="hashtag-pill">#{tag}</span>
          ))}
          {hashtags.length > 8 && (
            <span className="hashtag-pill" style={{opacity: 0.6}}>+{hashtags.length - 8} more</span>
          )}
        </div>
      )}
    </div>
  );
}
