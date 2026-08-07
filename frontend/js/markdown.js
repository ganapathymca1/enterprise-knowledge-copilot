/**
 * Minimal, safe markdown renderer for answer text.
 *
 * Why hand-rolled rather than a library: the answer text is untrusted output
 * from a language model, and pulling in a full markdown parser would mean
 * auditing its HTML sanitisation too. This renderer escapes *everything* first
 * and then re-introduces a deliberately tiny set of tags — bold, italics,
 * inline code, lists, tables and paragraphs. Raw HTML in the model's output can
 * never reach the DOM as markup.
 *
 * Citation markers like [2] are turned into focusable buttons so the
 * explainability panel can be opened straight from the sentence that used them.
 */

export function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function inline(text) {
  let out = escapeHtml(text);
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  // Citation markers -> interactive chips.
  out = out.replace(
    /\[(\d{1,2})\]/g,
    (_match, marker) =>
      `<button type="button" class="cite" data-marker="${marker}" ` +
      `aria-label="Show source ${marker}">${marker}</button>`
  );
  return out;
}

function isTableRow(line) {
  return /^\s*\|.*\|\s*$/.test(line);
}

function isDivider(line) {
  return /^\s*\|[\s:|-]+\|\s*$/.test(line);
}

function cells(line) {
  return line.trim().replace(/^\||\|$/g, '').split('|').map((cell) => cell.trim());
}

export function renderMarkdown(source) {
  const lines = String(source ?? '').replace(/\r\n/g, '\n').split('\n');
  const html = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];

    if (!line.trim()) { index += 1; continue; }

    // table
    if (isTableRow(line) && isTableRow(lines[index + 1] ?? '') && isDivider(lines[index + 1])) {
      const header = cells(line);
      index += 2;
      const body = [];
      while (index < lines.length && isTableRow(lines[index])) {
        body.push(cells(lines[index]));
        index += 1;
      }
      html.push(
        '<table><thead><tr>' +
          header.map((cell) => `<th>${inline(cell)}</th>`).join('') +
          '</tr></thead><tbody>' +
          body
            .map((row) => `<tr>${row.map((cell) => `<td>${inline(cell)}</td>`).join('')}</tr>`)
            .join('') +
          '</tbody></table>'
      );
      continue;
    }

    // lists
    const bullet = /^\s*[-*+]\s+(.*)$/.exec(line);
    const ordered = /^\s*\d+\.\s+(.*)$/.exec(line);
    if (bullet || ordered) {
      const tag = bullet ? 'ul' : 'ol';
      const pattern = bullet ? /^\s*[-*+]\s+(.*)$/ : /^\s*\d+\.\s+(.*)$/;
      const items = [];
      while (index < lines.length) {
        const match = pattern.exec(lines[index]);
        if (match) {
          items.push(match[1]);
          index += 1;
        } else if (/^\s{2,}\S/.test(lines[index]) && items.length) {
          items[items.length - 1] += ` ${lines[index].trim()}`;
          index += 1;
        } else {
          break;
        }
      }
      html.push(`<${tag}>${items.map((item) => `<li>${inline(item)}</li>`).join('')}</${tag}>`);
      continue;
    }

    // heading -> rendered as a bold lead line, since answers should not
    // introduce document structure into the transcript
    const heading = /^\s*#{1,6}\s+(.*)$/.exec(line);
    if (heading) {
      html.push(`<p><strong>${inline(heading[1])}</strong></p>`);
      index += 1;
      continue;
    }

    // paragraph (consume until a blank line or a block starts)
    const paragraph = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^\s*([-*+]|\d+\.)\s+/.test(lines[index]) &&
      !isTableRow(lines[index]) &&
      !/^\s*#{1,6}\s+/.test(lines[index])
    ) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    if (paragraph.length) html.push(`<p>${inline(paragraph.join(' '))}</p>`);
  }

  return html.join('');
}
