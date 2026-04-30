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
      <h2>Runtime Console</h2>
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
      <section class="inference-live">
        <div class="inference-metrics-grid">
          <article class="inference-card">
            <span class="inference-card__label">Current score</span>
            <strong class="inference-card__value" id="inf-current-score">0</strong>
          </article>
          <article class="inference-card">
            <span class="inference-card__label">Current max tile</span>
            <strong class="inference-card__value" id="inf-current-max-tile">0</strong>
          </article>
          <article class="inference-card">
            <span class="inference-card__label">Moves (episode)</span>
            <strong class="inference-card__value" id="inf-current-moves">0</strong>
          </article>
          <article class="inference-card">
            <span class="inference-card__label">Episodes complete</span>
            <strong class="inference-card__value" id="inf-episodes-complete">0</strong>
          </article>
        </div>

        <div class="inference-chart-grid">
          <article class="inference-card inference-card--chart">
            <span class="inference-card__label">Score vs move</span>
            <svg
              class="inference-chart"
              id="inf-score-chart"
              viewBox="0 0 360 140"
              role="img"
              aria-label="Score over move count"
            ></svg>
          </article>
          <article class="inference-card inference-card--chart">
            <span class="inference-card__label">Max tile vs move</span>
            <svg
              class="inference-chart"
              id="inf-max-tile-chart"
              viewBox="0 0 360 140"
              role="img"
              aria-label="Max tile over move count"
            ></svg>
          </article>
        </div>

        <article class="inference-card inference-card--chart">
          <div class="inference-controls">
            <span class="inference-card__label">Action distribution <span id="inf-action-sample">(N=0)</span></span>
            <div class="segmented-control" role="group" aria-label="Action distribution mode">
              <button type="button" class="segmented-control__button is-active" id="inf-mode-rolling">Rolling</button>
              <button type="button" class="segmented-control__button" id="inf-mode-cumulative">Cumulative</button>
            </div>
          </div>
          <p class="inference-note" id="inf-action-mode-note">Showing last 50 moves.</p>
          <div class="action-distribution" id="inf-action-distribution"></div>
        </article>

        <article class="inference-card">
          <span class="inference-card__label">Agent</span>
          <div class="inference-controls">
            <select class="segmented-control__select" id="inf-agent-select" aria-label="Agent selection">
              <option value="auto" selected>Auto (DQN → TD → Greedy)</option>
              <option value="dqn">DQN</option>
              <option value="td">TD n-tuple</option>
              <option value="greedy">Greedy (myopic)</option>
              <option value="nstep">MC lookahead (N-step)</option>
            </select>
            <button type="button" class="segmented-control__button is-active" id="inf-agent-start">Start</button>
            <button type="button" class="segmented-control__button" id="inf-agent-reset">Reset agent</button>
          </div>
          <p class="inference-note">Choose an agent, then press Start to connect.</p>
        </article>
      </section>
      <div class="payload-shell">
        <p class="payload-shell__label">Latest agent payload</p>
        <pre class="payload-shell__code"><code id="inference-payload-live">{escape(view.inference_payload)}</code></pre>
      </div>
    </section>
  </div>
</section>"""
