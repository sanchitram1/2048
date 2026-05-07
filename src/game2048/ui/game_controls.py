from __future__ import annotations


def render_game_controls() -> str:
    """Top dock: mode switch, match controls, and agent autoplay controls."""
    return """<section class="game-controls" aria-label="Gameplay and agent controls">
  <div class="game-controls__top">
    <div class="game-controls__title-block">
      <h2 class="game-controls__title">Play</h2>
      <p class="game-controls__subtitle">Choose a mode, agent, and start a session.</p>
    </div>
    <div class="game-controls__toolbar">
      <span class="fairness-badge fairness-badge--off" id="fairness-badge">Seed-matched start</span>
      <div class="mode-switch" role="group" aria-label="Gameplay mode">
        <button type="button" class="mode-switch__btn is-active" id="mode-play-against" data-game-mode="play_against">
          Play Against Agent
        </button>
        <button type="button" class="mode-switch__btn" id="mode-agent-autoplay" data-game-mode="agent_autoplay">
          Agent Autoplay
        </button>
      </div>
    </div>
  </div>

  <div class="game-controls__body">
    <div class="game-controls__col game-controls__col--meta">
      <p class="game-controls__hint" id="hint-play-against">
        Seed-matched start; trajectories diverge after your moves. If you reach game over first, click <strong>Continue Agent</strong> once to let the agent finish.
      </p>
      <p class="game-controls__note" id="game-controls-footnote">
        In <strong>Play Against Agent</strong>, press <strong>Start</strong> to open a match; <strong>Reset</strong> starts a new match.
      </p>

      <div class="match-summary" id="match-summary" hidden>
        <p class="match-summary__title" id="match-summary-title">Match</p>
        <p class="match-summary__body" id="match-summary-body"></p>
      </div>

    </div>

    <div class="game-controls__col game-controls__col--actions">
      <div class="game-controls__agent-row">
        <label class="game-controls__label" for="inf-agent-select">Agent</label>
        <div class="game-controls__agent-tools">
          <select class="agent-select--dark" id="inf-agent-select" aria-label="Agent selection">
            <option value="dqn" selected>DQN</option>
            <option value="td">TD n-tuple</option>
            <option value="greedy">Greedy (myopic)</option>
            <option value="nstep">MC lookahead (N-step)</option>
          </select>
          <button type="button" class="game-controls__btn game-controls__btn--primary" id="inf-agent-start">Start</button>
          <button type="button" class="game-controls__btn" id="inf-agent-stop">Stop</button>
          <button type="button" class="game-controls__btn" id="inf-agent-reset">Reset</button>
          <button type="button" class="game-controls__btn" id="inf-step-agent" hidden>Continue Agent</button>
        </div>
      </div>

      <div class="game-controls__speed" id="autoplay-speed-block">
        <label class="game-controls__label" for="inf-agent-speed">
          Agent step delay: <span id="inf-agent-speed-value">700</span> ms
        </label>
        <input
          type="range"
          id="inf-agent-speed"
          min="50"
          max="3000"
          step="50"
          value="700"
          aria-valuemin="50"
          aria-valuemax="3000"
          aria-valuenow="700"
          aria-label="Milliseconds between automatic agent steps"
        />
      </div>
    </div>

  </div>
</section>"""
