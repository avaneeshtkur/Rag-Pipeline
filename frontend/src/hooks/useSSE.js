export function useChatSSE() {
  const sendMessage = async (question, sessionId, videoA, videoB, onToken, onSources, onDone, onError) => {
    let response;
    try {
        response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                question: question,
                video_a_metadata: videoA,
                video_b_metadata: videoB
            })
        });
    } catch (err) {
        if (onError) onError('[Network error: Could not reach backend]');
        if (onDone) onDone();
        return;
    }

    if (!response.ok) {
        const text = await response.text();
        if (onError) onError(`[Server error ${response.status}: ${text}]`);
        if (onDone) onDone();
        return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');

        // Keep the last partial line in buffer
        buffer = lines.pop();

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || trimmed === 'data: [DONE]') continue;
            if (!trimmed.startsWith('data: ')) continue;

            const jsonStr = trimmed.slice(6).trim();
            if (!jsonStr) continue;

            try {
                const parsed = JSON.parse(jsonStr);
                if (parsed.type === 'token' && parsed.content) {
                    if (onToken) onToken(parsed.content);
                } else if (parsed.type === 'sources') {
                    if (onSources) onSources(parsed.content || []);
                } else if (parsed.type === 'done') {
                    if (onDone) onDone();
                    return;
                }
            } catch (e) {
                // Skip malformed line silently
            }
        }
    }

    // Stream ended without a done event — call onDone anyway
    if (onDone) onDone();
  };

  return { sendMessage };
}
