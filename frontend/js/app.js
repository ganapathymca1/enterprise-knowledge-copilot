/**
 * Enterprise Knowledge Copilot — frontend controller.
 *
 * No framework and no build step. That is a deliberate trade-off for this
 * submission: the brief requires the app to run from a clean environment with
 * the documented commands, and a zero-dependency frontend removes an entire
 * class of "it works on my machine" failure (node version, lockfile drift,
 * offline npm install). The code is still organised the way a component tree
 * would be — a small store, pure render helpers, and one place that talks to
 * the API — so porting it to React later is mechanical.
 *
 * Accessibility is treated as a requirement, not a polish item: the transcript
 * is a live region, streaming updates are announced without stealing focus,
 * every control is reachable and labelled, and the citation chips are real
 * buttons rather than styled spans.
 */

import {
  ApiError,
  getCorpus,
  getDirectory,
  getDocument,
  getHealth,
  listSessions,
  sendFeedback,
  streamChat,
} from './api.js';
import { escapeHtml, renderMarkdown } from './markdown.js';

const state = {
  sessionId: null,
  employeeId: localStorage.getItem('copilot.employeeId') || '',
  busy: false,
  documents: [],
  lastQuestion: '',
};

const el = {
  transcript: document.getElementById('transcript'),
  welcome: document.getElementById('welcome'),
  composer: document.getElementById('composer'),
  input: document.getElementById('composer-input'),
  sendBtn: document.getElementById('send-btn'),
  sendLabel: document.getElementById('send-label'),
  charCount: document.getElementById('char-count'),
  employeeSelect: document.getElementById('employee-select'),
  sessionList: document.getElementById('session-list'),
  docList: document.getElementById('doc-list'),
  corpusCount: document.getElementById('corpus-count'),
  statusDot: document.getElementById('status-dot'),
  statusText: document.getElementById('status-text'),
  statusNote: document.getElementById('status-note'),
  newChat: document.getElementById('new-chat'),
  toast: document.getElementById('toast'),
};

// --------------------------------------------------------------- utilities
function toast(message, ms = 3200) {
  el.toast.textContent = message;
  el.toast.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.toast.hidden = true; }, ms);
}

function scrollToBottom(force = false) {
  const node = el.transcript;
  const nearBottom = node.scrollHeight - node.scrollTop - node.clientHeight < 160;
  if (force || nearBottom) node.scrollTop = node.scrollHeight;
}

function formatMs(value) {
  return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${value} ms`;
}

// ------------------------------------------------------------ boot loading
async function boot() {
  wireEvents();
  await Promise.allSettled([loadHealth(), loadCorpus(), loadDirectory(), refreshSessions()]);
}

async function loadHealth() {
  try {
    const health = await getHealth();
    const degraded = health.status !== 'ok';
    el.statusDot.className = `dot ${degraded ? 'warn' : 'ok'}`;
    el.statusText.textContent = `${health.provider} · ${health.model}`;
    el.statusNote.textContent =
      health.notes[0] || `${health.documents} documents · ${health.chunks} passages indexed`;
  } catch (error) {
    el.statusDot.className = 'dot down';
    el.statusText.textContent = 'Backend unreachable';
    el.statusNote.textContent = error.message;
  }
}

async function loadCorpus() {
  try {
    const corpus = await getCorpus();
    state.documents = corpus.documents;
    el.corpusCount.textContent = `${corpus.documents.length} docs`;
    el.docList.replaceChildren(
      ...corpus.documents.map((doc) => {
        const item = document.createElement('li');
        const button = document.createElement('button');
        button.type = 'button';
        button.title = `${doc.doc_id} · v${doc.version} · effective ${doc.effective_date}`;
        button.innerHTML =
          `${escapeHtml(doc.title)}<span class="doc-meta">${escapeHtml(doc.doc_id)} · ` +
          `${doc.chunks} passages · v${escapeHtml(doc.version)}</span>`;
        button.addEventListener('click', () => openDocument(doc.doc_id));
        item.append(button);
        return item;
      })
    );
  } catch {
    el.corpusCount.textContent = 'unavailable';
  }
}

async function loadDirectory() {
  try {
    const people = await getDirectory();
    el.employeeSelect.replaceChildren(
      ...people.map((person) => {
        const option = document.createElement('option');
        option.value = person.employee_id;
        option.textContent = `${person.full_name} — ${person.job_title}`;
        return option;
      })
    );
    if (state.employeeId && people.some((p) => p.employee_id === state.employeeId)) {
      el.employeeSelect.value = state.employeeId;
    } else {
      state.employeeId = el.employeeSelect.value;
    }
  } catch {
    el.employeeSelect.replaceChildren(new Option('Directory unavailable', ''));
  }
}

async function refreshSessions() {
  try {
    const sessions = await listSessions();
    el.sessionList.replaceChildren(
      ...sessions.slice(0, 12).map((session) => {
        const item = document.createElement('li');
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = session.title;
        button.title = `${session.turns} question(s)`;
        if (session.session_id === state.sessionId) button.setAttribute('aria-current', 'true');
        // Server-side history is kept for the *model*; the transcript in the
        // page is the source of truth for the user, so switching sessions only
        // changes which conversation new turns are appended to.
        button.addEventListener('click', () => {
          state.sessionId = session.session_id;
          refreshSessions();
          toast('Continuing that conversation. New answers will use its history.');
          el.input.focus();
        });
        item.append(button);
        return item;
      })
    );
  } catch {
    /* the session list is a convenience; failing to load it must not break chat */
  }
}

// ----------------------------------------------------------------- sending
function wireEvents() {
  el.composer.addEventListener('submit', (event) => {
    event.preventDefault();
    submitQuestion(el.input.value);
  });

  el.input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submitQuestion(el.input.value);
    }
  });

  el.input.addEventListener('input', () => {
    el.input.style.height = 'auto';
    el.input.style.height = `${Math.min(el.input.scrollHeight, 180)}px`;
    el.charCount.textContent = `${el.input.value.length} / 1200`;
  });

  el.employeeSelect.addEventListener('change', () => {
    state.employeeId = el.employeeSelect.value;
    localStorage.setItem('copilot.employeeId', state.employeeId);
    toast('Identity changed. Record lookups now use this employee.');
  });

  el.newChat.addEventListener('click', () => {
    state.sessionId = null;
    el.transcript.replaceChildren(el.welcome);
    el.welcome.hidden = false;
    refreshSessions();
    el.input.focus();
  });

  for (const button of document.querySelectorAll('.example')) {
    button.addEventListener('click', () => submitQuestion(button.textContent.trim()));
  }
}

async function submitQuestion(rawText) {
  const text = rawText.trim();
  if (!text || state.busy) return;

  state.lastQuestion = text;
  el.welcome.hidden = true;
  el.input.value = '';
  el.input.style.height = 'auto';
  el.charCount.textContent = '0 / 1200';

  appendUserMessage(text);
  const pending = appendPendingMessage();
  setBusy(true);

  let answerText = '';
  let sources = null;

  streamChat(
    { message: text, session_id: state.sessionId },
    state.employeeId,
    {
      onStatus: (data) => pending.setStatus(data.label),
      onSources: (data) => {
        sources = data;
        state.sessionId = data.session_id;
      },
      onDelta: (piece) => {
        answerText += piece;
        pending.setAnswer(answerText);
      },
      onDone: (payload) => {
        setBusy(false);
        pending.finalise({ ...payload, answer: answerText, citations: sources?.citations ?? payload.citations ?? [] });
        refreshSessions();
      },
      onError: (error) => {
        setBusy(false);
        pending.fail(error);
      },
    }
  );
}

function setBusy(busy) {
  state.busy = busy;
  el.sendBtn.disabled = busy;
  el.input.disabled = busy;
  el.sendLabel.textContent = busy ? 'Working…' : 'Ask';
  if (!busy) el.input.focus();
}

// ---------------------------------------------------------------- messages
function appendUserMessage(text) {
  const node = document.getElementById('tpl-user-message').content.cloneNode(true);
  node.querySelector('.bubble').textContent = text;
  el.transcript.append(node);
  scrollToBottom(true);
}

function appendPendingMessage() {
  const fragment = document.getElementById('tpl-assistant-message').content.cloneNode(true);
  const article = fragment.querySelector('.msg-assistant');
  const answer = article.querySelector('.answer');
  const badges = article.querySelector('.badges');
  const actions = article.querySelector('.msg-actions');

  actions.hidden = true;
  answer.innerHTML =
    '<div class="thinking"><span class="spinner"></span><span class="stage">Checking your question</span></div>' +
    '<div class="skeleton"><span></span><span></span><span></span></div>';

  el.transcript.append(fragment);
  scrollToBottom(true);

  return {
    node: article,
    setStatus(label) {
      const stage = answer.querySelector('.stage');
      if (stage) stage.textContent = label;
    },
    setAnswer(text) {
      answer.innerHTML = renderMarkdown(text);
      scrollToBottom();
    },
    fail(error) {
      answer.innerHTML = '';
      const box = document.createElement('div');
      box.className = 'error-box';
      box.append(document.createTextNode(error.message));
      if (error.retryable !== false) {
        const retry = document.createElement('button');
        retry.type = 'button';
        retry.textContent = 'Try again';
        retry.addEventListener('click', () => {
          article.remove();
          submitQuestion(state.lastQuestion);
        });
        box.append(document.createElement('br'), retry);
      }
      answer.append(box);
      badges.replaceChildren(badge('error', 'error'));
      scrollToBottom();
    },
    finalise(payload) {
      answer.innerHTML = renderMarkdown(payload.answer);
      renderBadges(badges, payload);
      renderNotices(article, payload.notices ?? []);
      renderRecordCards(article, payload.tool_calls ?? []);
      renderFollowups(article, payload.followups ?? []);
      renderEvidence(article, payload);
      wireFeedback(article, payload);
      wireCitationChips(article, payload.citations ?? []);
      actions.hidden = false;
      scrollToBottom();
    },
  };
}

function badge(kind, label, title = '') {
  const span = document.createElement('span');
  span.className = `badge badge-${kind}`;
  span.textContent = label;
  if (title) span.title = title;
  return span;
}

const ANSWER_TYPE_LABEL = {
  grounded: 'from policy',
  tool: 'from your records',
  hybrid: 'records + policy',
  abstained: 'no answer found',
  refused: 'redirected',
  error: 'error',
};

const CONFIDENCE_TITLE = {
  high: 'Strong retrieval match and every checkable claim is cited.',
  medium: 'Reasonable match, or some claims are not individually cited. Check the sources.',
  low: 'Weak match or poorly attributed. Treat this as a pointer, not an answer.',
};

function renderBadges(container, payload) {
  const items = [
    badge(payload.answer_type, ANSWER_TYPE_LABEL[payload.answer_type] ?? payload.answer_type),
  ];
  if (payload.confidence) {
    items.push(
      badge(
        `conf-${payload.confidence}`,
        `${payload.confidence} confidence`,
        CONFIDENCE_TITLE[payload.confidence] ?? ''
      )
    );
  }
  container.replaceChildren(...items);
}

function renderNotices(article, notices) {
  const list = article.querySelector('.notices');
  if (!notices.length) { list.hidden = true; return; }
  list.replaceChildren(
    ...notices.map((text) => {
      const item = document.createElement('li');
      item.textContent = text;
      return item;
    })
  );
  list.hidden = false;
}

function renderFollowups(article, followups) {
  const container = article.querySelector('.followups');
  if (!followups.length) { container.hidden = true; return; }
  container.replaceChildren(
    ...followups.map((text) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'followup';
      button.textContent = text;
      button.addEventListener('click', () => submitQuestion(text));
      return button;
    })
  );
  container.hidden = false;
}

/** Structured tool output is rendered as a record card, not left to prose. */
function renderRecordCards(article, toolCalls) {
  const container = article.querySelector('.record-cards');
  container.replaceChildren();
  for (const call of toolCalls) {
    if (!call.ok || !call.summary) continue;
    const card = document.createElement('div');
    card.className = 'record-card';
    const heading = document.createElement('h4');
    heading.textContent = `${call.name.replaceAll('_', ' ')} · ${call.latency_ms} ms`;
    const body = document.createElement('div');
    body.innerHTML = renderMarkdown(call.summary);
    card.append(heading, body);
    container.append(card);
  }
}

function renderEvidence(article, payload) {
  const details = article.querySelector('.evidence');
  const body = article.querySelector('.evidence-body');
  const citations = payload.citations ?? [];
  const toolCalls = payload.tool_calls ?? [];
  if (!citations.length && !toolCalls.length) { details.hidden = true; return; }

  article.querySelector('.evidence-label').textContent =
    `Sources & reasoning (${citations.length} passage${citations.length === 1 ? '' : 's'})`;

  const used = new Set(
    [...String(payload.answer ?? '').matchAll(/\[(\d{1,2})\]/g)].map((match) => Number(match[1]))
  );

  const parts = [];

  for (const call of toolCalls) {
    const node = document.createElement('div');
    node.className = `tool-call${call.ok ? '' : ' failed'}`;
    node.innerHTML =
      `<strong>Tool</strong> <code>${escapeHtml(call.name)}(${escapeHtml(
        JSON.stringify(call.arguments ?? {})
      )})</code> — ${call.ok ? 'returned data' : escapeHtml(call.error ?? 'failed')} in ${call.latency_ms} ms`;
    parts.push(node);
  }

  for (const citation of citations) {
    const node = document.createElement('div');
    node.className = `source${used.has(citation.marker) ? ' is-used' : ''}`;
    node.id = `source-${payload.trace_id}-${citation.marker}`;
    node.tabIndex = -1;
    const percent = Math.round(Math.min(1, citation.score) * 100);
    node.innerHTML = `
      <div class="source-head">
        <span class="source-title">${escapeHtml(citation.title)}${
          citation.section ? ` — ${escapeHtml(citation.section)}` : ''
        }</span>
        <span class="source-marker">${citation.marker}</span>
      </div>
      <p class="source-meta">${escapeHtml(citation.doc_id)} · v${escapeHtml(
        citation.version
      )} · effective ${escapeHtml(citation.effective_date)} · owner ${escapeHtml(citation.owner)}</p>
      <p class="source-snippet">${escapeHtml(citation.snippet)}</p>
      <div class="score-bar"><span style="width:${percent}%"></span></div>
      <div class="score-label"><span>retrieval relevance</span><span>${percent}%</span></div>
    `;
    const open = document.createElement('button');
    open.type = 'button';
    open.className = 'source-open';
    open.textContent = 'Open full policy →';
    open.addEventListener('click', () => openDocument(citation.doc_id, citation.section));
    node.append(open);
    parts.push(node);
  }

  const trace = document.createElement('dl');
  trace.className = 'trace-grid';
  const rows = [
    ['Provider', `${payload.provider}${payload.model ? ` · ${payload.model}` : ''}`],
    ['Grounding', `${Math.round((payload.grounding_score ?? 0) * 100)}%`],
    ['Total', formatMs(payload.timings?.total_ms ?? 0)],
    ['Retrieval', formatMs(payload.timings?.retrieval_ms ?? 0)],
    ['Generation', formatMs(payload.timings?.generation_ms ?? 0)],
    ['Tools', formatMs(payload.timings?.tool_ms ?? 0)],
  ];
  if (payload.rewritten_query) rows.push(['Search query used', payload.rewritten_query]);
  rows.push(['Trace id', payload.trace_id]);
  for (const [label, value] of rows) {
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.textContent = value;
    trace.append(dt, dd);
  }
  parts.push(trace);

  body.replaceChildren(...parts);
  details.hidden = false;
}

/** Clicking [2] in the answer opens the panel and moves focus to that source. */
function wireCitationChips(article, citations) {
  const details = article.querySelector('.evidence');
  for (const chip of article.querySelectorAll('.cite')) {
    chip.addEventListener('click', () => {
      const marker = Number(chip.dataset.marker);
      const citation = citations.find((item) => item.marker === marker);
      if (!citation) return;
      details.open = true;
      const target = details.querySelector(`.source-marker`)?.closest('.source');
      const node = [...details.querySelectorAll('.source')].find(
        (candidate) => candidate.querySelector('.source-marker').textContent === String(marker)
      ) ?? target;
      node?.scrollIntoView({ block: 'nearest' });
      node?.focus();
    });
  }
}

// ---------------------------------------------------------------- feedback
const REASONS = [
  ['incorrect', 'Incorrect'],
  ['incomplete', 'Incomplete'],
  ['outdated', 'Out of date'],
  ['wrong_source', 'Wrong source'],
  ['unclear', 'Unclear'],
  ['not_relevant', 'Not relevant'],
];

function wireFeedback(article, payload) {
  const form = article.querySelector('.feedback-form');
  const thanks = article.querySelector('.vote-thanks');
  const buttons = [...article.querySelectorAll('.vote-btn')];
  let selectedReason = 'other';

  const reasons = form.querySelector('.reasons');
  reasons.replaceChildren(
    ...REASONS.map(([value, label]) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'reason-btn';
      button.textContent = label;
      button.setAttribute('aria-pressed', 'false');
      button.addEventListener('click', () => {
        selectedReason = value;
        for (const other of reasons.children) other.setAttribute('aria-pressed', 'false');
        button.setAttribute('aria-pressed', 'true');
      });
      return button;
    })
  );

  const submit = async (rating, reason, comment) => {
    try {
      const result = await sendFeedback({
        trace_id: payload.trace_id,
        session_id: payload.session_id,
        rating,
        reason,
        comment,
      });
      thanks.textContent =
        rating === 'up'
          ? 'Thanks — logged against this answer.'
          : `Thanks — routed to ${result.routed_to} for review.`;
      thanks.hidden = false;
      form.hidden = true;
    } catch (error) {
      toast(error.message);
    }
  };

  for (const button of buttons) {
    button.addEventListener('click', () => {
      const rating = button.dataset.rating;
      for (const other of buttons) other.setAttribute('aria-pressed', String(other === button));
      if (rating === 'up') {
        submit('up', 'helpful', '');
      } else {
        form.hidden = false;
        form.querySelector('.feedback-comment').focus();
      }
    });
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    submit('down', selectedReason, form.querySelector('.feedback-comment').value.trim());
  });

  form.querySelector('.cancel-feedback').addEventListener('click', () => {
    form.hidden = true;
    for (const button of buttons) button.setAttribute('aria-pressed', 'false');
  });
}

// ----------------------------------------------------------- document view
async function openDocument(docId, section = '') {
  let doc;
  try {
    doc = await getDocument(docId);
  } catch (error) {
    toast(error instanceof ApiError ? error.message : 'Could not open that document.');
    return;
  }

  const overlay = document.createElement('div');
  overlay.className = 'doc-viewer';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');
  overlay.setAttribute('aria-label', doc.title);

  const panel = document.createElement('div');
  panel.className = 'doc-viewer-panel';

  const head = document.createElement('div');
  head.className = 'doc-viewer-head';
  head.innerHTML = `<div><h3>${escapeHtml(doc.title)}</h3>
    <p class="hint">${escapeHtml(doc.doc_id)} · v${escapeHtml(doc.version)} ·
    effective ${escapeHtml(doc.effective_date)} · owner ${escapeHtml(doc.owner)}</p></div>`;

  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'ghost-btn';
  close.textContent = 'Close';
  head.append(close);

  const pre = document.createElement('pre');
  pre.textContent = doc.body;
  panel.append(head, pre);
  overlay.append(panel);
  document.body.append(overlay);

  // Highlight the cited section heading so the citation lands in context.
  if (section) {
    const needle = section.split('›').pop().trim();
    const position = doc.body.indexOf(needle);
    if (position >= 0) {
      pre.innerHTML =
        escapeHtml(doc.body.slice(0, position)) +
        `<mark id="cited-section">${escapeHtml(needle)}</mark>` +
        escapeHtml(doc.body.slice(position + needle.length));
      panel.querySelector('#cited-section')?.scrollIntoView({ block: 'center' });
    }
  }

  const previouslyFocused = document.activeElement;
  const dismiss = () => {
    overlay.remove();
    document.removeEventListener('keydown', onKey);
    previouslyFocused?.focus?.();
  };
  const onKey = (event) => { if (event.key === 'Escape') dismiss(); };

  close.addEventListener('click', dismiss);
  overlay.addEventListener('click', (event) => { if (event.target === overlay) dismiss(); });
  document.addEventListener('keydown', onKey);
  close.focus();
}

boot();
