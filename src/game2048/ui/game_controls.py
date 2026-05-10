from __future__ import annotations


def render_game_controls() -> str:
    """Top dock: mode switch, match controls, and agent autoplay controls.

    Layout references checked against repo screenshots:
    - assets/versus-mode.png — numbered steps + right-hand controls.
    - assets/agent-mode.png — tagline + orange slider under thumb + controls.
    """
    return """<section class="game-controls" aria-label="Gameplay and agent controls">
  <div class="game-controls__body">
    <div class="game-controls__col game-controls__col--meta">
      <div class="game-controls__hero-shell">
        <div class="game-controls__hero-head">
          <h2 class="game-controls__title game-controls__title--hero">Are you smarter than an DQN RL agent?</h2>
        </div>

        <div id="versus-copy" class="game-controls__mode-copy">
          <ol class="game-controls__steps" id="versus-steps-static">
            <li>Choose your opponent</li>
            <li>Press Start</li>
          </ol>
        </div>

        <div id="autoplay-copy" class="game-controls__mode-copy" hidden>
          <p class="game-controls__tagline">Drag to watch the RL agent rip it</p>
          <div class="game-controls__speed game-controls__speed--left" id="autoplay-speed-block">
            <div class="game-controls__speed-shell">
              <input
                type="range"
                id="inf-agent-speed"
                min="50"
                max="500"
                step="50"
                value="500"
                aria-valuemin="50"
                aria-valuemax="500"
                aria-valuenow="500"
                aria-label="Milliseconds between automatic agent steps"
              />
              <span id="inf-agent-speed-value" class="game-controls__speed-value-under-thumb" aria-live="polite">500ms</span>
            </div>
          </div>
        </div>
      </div>

      <div class="match-summary" id="match-summary" hidden>
        <p class="match-summary__title" id="match-summary-title">Match</p>
        <p class="match-summary__body" id="match-summary-body"></p>
      </div>

    </div>

    <div class="game-controls__col game-controls__col--actions">
      <div class="game-controls__agent-stack">
        <div class="mode-switch" role="group" aria-label="Gameplay mode">
          <button type="button" class="mode-switch__btn is-active" id="mode-play-against" data-game-mode="play_against">
            Versus
          </button>
          <button type="button" class="mode-switch__btn" id="mode-agent-autoplay" data-game-mode="agent_autoplay">
            Autoplay
          </button>
        </div>

        <label class="game-controls__label game-controls__label--sr" for="inf-agent-select">Agent</label>
        <select class="agent-select--dark" id="inf-agent-select" aria-label="Agent selection">
          <option value="dqn" selected>DQN</option>
          <option value="td">TD n-tuple</option>
          <option value="greedy">Greedy (myopic)</option>
          <option value="nstep">MC lookahead (N-step)</option>
        </select>
        <div class="game-controls__agent-btn-row">
          <button type="button" class="game-controls__btn game-controls__btn--primary" id="inf-agent-start">Start</button>
          <button type="button" class="game-controls__btn" id="inf-agent-stop">Stop</button>
          <button type="button" class="game-controls__btn" id="inf-agent-reset">Reset</button>
        </div>
        <button type="button" class="game-controls__btn game-controls__btn--stretch" id="inf-step-agent" hidden>Continue Agent</button>
      </div>
    </div>

  </div>
</section>"""
