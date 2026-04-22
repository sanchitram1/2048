from __future__ import annotations

from html import escape

from game2048.ui.models import TerminalView


def _render_log_line(line: str) -> str:
    if line.startswith("[") and "]" in line:
        channel, message = line[1:].split("]", maxsplit=1)
        return (
            '<div class="terminal-line">'
            f'<span class="terminal-line__channel">{escape(channel)}</span>'
            f"<span>{escape(message.strip())}</span>"
            "</div>"
        )

    return f'<div class="terminal-line"><span>{escape(line)}</span></div>'


def render_terminal(view: TerminalView) -> str:
    lines_html = "".join(_render_log_line(line) for line in view.log_lines)
    cards_html = "".join(
        """
<article class="inference-card">
  <span class="inference-card__label">{label}</span>
  <strong class="inference-card__value">{value}</strong>
</article>""".format(
            label=escape(card.label),
            value=escape(card.value),
        )
        for card in view.inference_cards
    )

    return f"""<section class="terminal-shell">
  <header class="terminal-shell__header">
    <div>
      <p class="eyebrow">Milestone 03</p>
      <h2>Runtime Console</h2>
      <p class="terminal-shell__subtitle">Logs, agent telemetry, and score comparisons live together down here.</p>
    </div>
    <nav class="tab-nav" aria-label="runtime console tabs">
      <button class="tab-nav__button is-active" type="button" data-tab-trigger="terminal">Terminal</button>
      <button class="tab-nav__button" type="button" data-tab-trigger="inference">Inference</button>
    </nav>
  </header>
  <div class="tab-panels">
    <section class="tab-panel is-active" data-tab-panel="terminal">
      <div class="terminal-window" id="terminal-log">{lines_html}</div>
    </section>
    <section class="tab-panel" data-tab-panel="inference" hidden>
      <div class="inference-grid">{cards_html}</div>
      <div class="payload-shell">
        <p class="payload-shell__label">Mock event contract</p>
        <pre class="payload-shell__code"><code>{escape(view.inference_payload)}</code></pre>
      </div>
    </section>
  </div>
</section>"""
