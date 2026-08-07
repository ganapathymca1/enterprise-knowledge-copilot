/**
 * Thin API client.
 *
 * Two things it does beyond `fetch`:
 *  - normalises every failure into one `ApiError` shape, so the UI has a single
 *    error component instead of a different message per endpoint;
 *  - parses the SSE stream by hand, because the chat stream is a POST and
 *    `EventSource` only speaks GET.
 */

export class ApiError extends Error {
  constructor(message, { status = 0, retryable = false, detail = '' } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.retryable = retryable;
    this.detail = detail;
  }
}

const JSON_HEADERS = { 'Content-Type': 'application/json' };

async function toApiError(response) {
  let detail = '';
  let retryable = response.status >= 500 || response.status === 429;
  try {
    const body = await response.json();
    const payload = body?.detail ?? body;
    detail = typeof payload === 'string' ? payload : payload?.detail || payload?.error || '';
    if (typeof payload === 'object' && payload && 'retryable' in payload) {
      retryable = Boolean(payload.retryable);
    }
  } catch {
    detail = response.statusText;
  }
  const message =
    response.status === 429
      ? 'The copilot is rate limited right now.'
      : response.status >= 500
        ? 'The copilot backend had a problem answering that.'
        : detail || `Request failed (${response.status}).`;
  return new ApiError(message, { status: response.status, retryable, detail });
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (cause) {
    throw new ApiError('Could not reach the copilot backend. Is it running?', {
      retryable: true,
      detail: String(cause),
    });
  }
  if (!response.ok) throw await toApiError(response);
  return response;
}

export async function getHealth() {
  return (await request('/api/health')).json();
}

export async function getCorpus() {
  return (await request('/api/corpus')).json();
}

export async function getDocument(docId) {
  return (await request(`/api/documents/${encodeURIComponent(docId)}`)).json();
}

export async function getDirectory() {
  return (await request('/api/directory')).json();
}

export async function listSessions() {
  return (await request('/api/sessions')).json();
}

export async function sendFeedback(payload) {
  const response = await request('/api/feedback', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
  return response.json();
}

/**
 * Stream a chat turn.
 *
 * `handlers` receives: onStatus, onSources, onDelta, onDone, onError.
 * Returns an abort function so the caller can cancel an in-flight turn.
 */
export function streamChat(payload, employeeId, handlers) {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: employeeId ? { ...JSON_HEADERS, 'X-Employee-Id': employeeId } : JSON_HEADERS,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!response.ok) throw await toApiError(response);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line.
        let boundary = buffer.indexOf('\n\n');
        while (boundary !== -1) {
          const frame = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          dispatch(frame, handlers);
          boundary = buffer.indexOf('\n\n');
        }
      }
      handlers.onClose?.();
    } catch (error) {
      if (error.name === 'AbortError') return;
      handlers.onError?.(
        error instanceof ApiError
          ? error
          : new ApiError('The connection to the copilot dropped.', { retryable: true })
      );
    }
  })();

  return () => controller.abort();
}

function dispatch(frame, handlers) {
  let event = 'message';
  const dataLines = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return;

  let data;
  try {
    data = JSON.parse(dataLines.join('\n'));
  } catch {
    return;
  }

  switch (event) {
    case 'status': handlers.onStatus?.(data); break;
    case 'sources': handlers.onSources?.(data); break;
    case 'delta': handlers.onDelta?.(data.text ?? ''); break;
    case 'done': handlers.onDone?.(data); break;
    case 'error':
      handlers.onError?.(
        new ApiError(data.detail || 'The copilot could not answer that.', {
          retryable: Boolean(data.retryable),
        })
      );
      break;
    default: break;
  }
}
