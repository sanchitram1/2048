from __future__ import annotations

from textwrap import dedent


def render_styles() -> str:
    return dedent(
        """
        :root {
          color-scheme: light;
          --page-bg: #efd4a7;
          --page-bg-deep: #d1a268;
          --panel: rgba(43, 31, 24, 0.94);
          --panel-soft: rgba(70, 50, 38, 0.82);
          --panel-border: rgba(255, 241, 223, 0.16);
          --text-main: #fff4e6;
          --text-muted: rgba(255, 244, 230, 0.72);
          --shadow: 0 24px 60px rgba(60, 32, 7, 0.22);
          --surface: rgba(255, 243, 226, 0.10);
          --surface-strong: rgba(255, 243, 226, 0.16);
          --font-display: "Avenir Next Condensed", "Futura", "Trebuchet MS", sans-serif;
          --font-body: "Avenir Next", "Segoe UI", sans-serif;
          --font-mono: "SFMono-Regular", "Menlo", "Monaco", monospace;
          --terminal-log-height: min(260px, 32vh);

          /*
           * Human / Agent board panel — ONE vertical knob (full width unchanged).
           * 1 = default; lower = shorter panel (tile height + vertical padding/gaps scale down).
           * Try 0.85–0.95 to shave ~50–120px depending on viewport.
           */
          --board-panel-height-scale: 0.75;
        }

        * {
          box-sizing: border-box;
        }

        body {
          margin: 0;
          min-height: 100vh;
          font-family: var(--font-body);
          color: var(--text-main);
          background:
            radial-gradient(circle at top, rgba(255, 245, 224, 0.65), transparent 35%),
            linear-gradient(180deg, #f5dfb9 0%, var(--page-bg) 45%, var(--page-bg-deep) 100%);
        }

        body::before {
          content: "";
          position: fixed;
          inset: 0;
          pointer-events: none;
          background:
            linear-gradient(90deg, rgba(255, 255, 255, 0.07) 1px, transparent 1px),
            linear-gradient(rgba(90, 55, 23, 0.04) 1px, transparent 1px);
          background-size: 24px 24px;
          opacity: 0.4;
        }

        .app-shell {
          position: relative;
          max-width: 1440px;
          margin: 0 auto;
          padding: 20px 20px 28px;
        }

        .top-band {
          display: grid;
          gap: 16px;
          margin-bottom: 16px;
          align-items: stretch;
        }

        @media (min-width: 1040px) {
          .top-band {
            grid-template-columns: minmax(0, 1fr);
            gap: 18px 22px;
            margin-bottom: 14px;
          }
        }

        .hero {
          display: flex;
          margin-bottom: 0;
        }

        .hero__panel,
        .board-card,
        .terminal-shell {
          position: relative;
          overflow: hidden;
          border: 1px solid var(--panel-border);
          border-radius: 28px;
          background: var(--panel);
          box-shadow: var(--shadow);
          backdrop-filter: blur(12px);
        }

        .hero__panel {
          padding: 16px 18px 18px;
          width: 100%;
          height: 100%;
        }

        .hero__panel::after,
        .board-card::after,
        .terminal-shell::after {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(140deg, rgba(255, 255, 255, 0.08), transparent 55%);
          pointer-events: none;
        }

        .eyebrow {
          margin: 0 0 8px;
          text-transform: uppercase;
          letter-spacing: 0.22em;
          font-size: 0.72rem;
          color: rgba(255, 231, 194, 0.75);
        }

        .hero h1,
        .hero h2,
        .board-card h2,
        .terminal-shell h2 {
          margin: 0;
          font-family: var(--font-display);
          letter-spacing: 0.02em;
        }

        .hero h1 {
          font-size: clamp(1.9rem, 3vw, 2.9rem);
          line-height: 1.04;
          max-width: none;
        }

        .hero__lede,
        .lane-tools__hint {
          margin: 8px 0 0;
          color: var(--text-muted);
          line-height: 1.4;
          font-size: 0.82rem;
        }

        .game-controls {
          position: relative;
          overflow: hidden;
          margin-bottom: 0;
          padding: 14px 16px 16px;
          border: 1px solid var(--panel-border);
          border-radius: 22px;
          background: var(--panel);
          box-shadow: var(--shadow);
          backdrop-filter: blur(12px);
          font-family: var(--font-body);
          font-size: 1.05rem;
        }

        .game-controls::after {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(140deg, rgba(255, 255, 255, 0.06), transparent 55%);
          pointer-events: none;
        }

        .game-controls__mode-copy[hidden],
        #autoplay-copy[hidden],
        #versus-copy[hidden] {
          display: none !important;
        }

        .game-controls__hero-shell {
          padding: 0;
          margin: 0;
          background: transparent;
        }

        .game-controls__hero-head {
          width: 100%;
          overflow-x: auto;
          scrollbar-width: none;
          margin-bottom: 4px;
        }

        .game-controls__hero-head::-webkit-scrollbar {
          height: 0;
        }

        .game-controls__title {
          margin: 0;
          font-family: var(--font-display);
          font-weight: 400;
        }

        /* Match `.board-card h2` sizing and rhythm */
        .game-controls__title--hero {
          display: inline-block;
          white-space: nowrap;
          margin: 0;
          padding-right: 4px;
          font-size: clamp(1.28rem, 1.75vw, 1.6rem);
          line-height: 1.12;
          letter-spacing: 0.02em;
          font-weight: 700;
          color: var(--text-main);
        }

        /* Match board body copy / stat line rhythm */
        .game-controls__steps {
          margin: 12px 0 0;
          padding-left: 1.35em;
          font-family: var(--font-body);
          font-size: 1rem;
          line-height: 1.45;
          font-weight: 400;
          color: var(--text-main);
        }

        .game-controls__steps li {
          padding-left: 0.12em;
          font-weight: 400;
        }

        .game-controls__steps li::marker {
          color: rgba(255, 233, 207, 0.55);
          font-weight: 400;
        }

        .game-controls__tagline {
          margin: 12px 0 14px;
          font-family: var(--font-body);
          font-size: 1rem;
          line-height: 1.45;
          color: var(--text-main);
        }

        .game-controls .mode-switch {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 6px;
          width: 100%;
          padding: 4px;
          margin: 0;
          border-radius: 14px;
          background: rgba(255, 255, 255, 0.08);
          box-sizing: border-box;
        }

        .game-controls .mode-switch__btn {
          margin: 0;
          padding: 8px 12px;
          border: 0;
          border-radius: 10px;
          background: transparent;
          color: rgba(236, 243, 255, 0.78);
          font: inherit;
          font-size: 0.8rem;
          font-weight: 600;
          letter-spacing: 0.02em;
          cursor: pointer;
        }

        .game-controls .mode-switch__btn.is-active {
          background: rgba(238, 165, 94, 0.28);
          color: #fff8ec;
        }

        .game-controls__speed-shell {
          position: relative;
          padding: 0 0 26px;
          margin-top: 4px;
          background: transparent;
        }

        .game-controls__speed-shell #inf-agent-speed {
          width: 100%;
          accent-color: rgba(238, 165, 94, 0.85);
          cursor: pointer;
        }

        .game-controls__speed-value-under-thumb {
          position: absolute;
          bottom: 0;
          left: 100%;
          transform: translateX(-50%);
          font-family: var(--font-body);
          font-size: 1rem;
          font-weight: 600;
          color: var(--text-main);
          pointer-events: none;
        }

        .game-controls__body {
          display: grid;
          gap: 12px;
          align-items: start;
        }

        @media (min-width: 720px) {
          .game-controls__body {
            grid-template-columns: minmax(0, 13fr) minmax(0, 7fr);
            gap: 12px 24px;
            align-items: start;
          }

          .game-controls__col--actions {
            display: flex;
            justify-content: center;
            align-items: flex-start;
          }

          .game-controls__note--span {
            grid-column: 1 / -1;
          }
        }

        .game-controls__col--meta,
        .game-controls__col--actions {
          min-width: 0;
        }

        .match-summary {
          margin-top: 8px;
          margin-bottom: 0;
          padding: 10px 12px;
          border-radius: 14px;
          background: rgba(255, 255, 255, 0.07);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .match-summary__title {
          margin: 0 0 6px;
          font-size: 0.72rem;
          text-transform: uppercase;
          letter-spacing: 0.16em;
          color: rgba(255, 233, 207, 0.62);
        }

        .match-summary__body {
          margin: 0;
          font-size: 0.95rem;
          line-height: 1.45;
          color: #fff6eb;
        }

        .game-controls__row {
          margin-bottom: 0;
        }

        .game-controls__row--seed {
          max-width: none;
        }

        .game-controls__label {
          display: block;
          font-size: 0.74rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          font-weight: 600;
          color: rgba(255, 233, 207, 0.62);
          margin-bottom: 8px;
        }

        .game-controls__label--sr {
          position: absolute;
          width: 1px;
          height: 1px;
          padding: 0;
          margin: -1px;
          overflow: hidden;
          clip: rect(0, 0, 0, 0);
          clip-path: inset(50%);
          white-space: nowrap;
          border: 0;
        }

        .game-controls__input {
          width: 100%;
          padding: 10px 12px;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.14);
          background: rgba(12, 16, 24, 0.35);
          color: var(--text-main);
          font: inherit;
          font-size: 0.9rem;
        }

        .game-controls__input::placeholder {
          color: rgba(255, 244, 230, 0.45);
        }

        .game-controls__agent-stack {
          position: relative;
          display: flex;
          flex-direction: column;
          gap: 10px;
          width: 100%;
          box-sizing: border-box;
        }

        @media (min-width: 720px) {
          .game-controls__agent-stack {
            max-width: min(320px, 100%);
          }
        }

        .game-controls__agent-btn-row {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 8px;
          align-items: stretch;
        }

        .game-controls__agent-btn-row .game-controls__btn {
          padding-left: 8px;
          padding-right: 8px;
          text-align: center;
        }

        .game-controls__btn--stretch {
          width: 100%;
          margin-top: 2px;
        }

        #inf-step-agent[hidden] {
          display: none !important;
        }

        .game-controls .agent-select--dark {
          width: 100%;
          min-width: 0;
          border: 1px solid rgba(20, 24, 32, 0.35);
          background: #f8fafc;
          color: #111827;
          font: inherit;
          font-size: 0.88rem;
          font-weight: 500;
          padding: 10px 12px;
          border-radius: 12px;
          cursor: pointer;
          box-sizing: border-box;
        }

        .agent-select--dark option {
          color: #111827;
          background: #f8fafc;
        }

        .game-controls .game-controls__btn {
          border: 1px solid rgba(255, 255, 255, 0.16);
          background: rgba(255, 255, 255, 0.1);
          color: rgba(255, 248, 236, 0.92);
          font: inherit;
          font-size: 0.82rem;
          font-weight: 600;
          padding: 10px 16px;
          border-radius: 12px;
          cursor: pointer;
        }

        .game-controls .game-controls__btn--primary {
          background: rgba(207, 107, 45, 0.45);
          border-color: rgba(255, 200, 160, 0.35);
        }

        .board-layout {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 18px;
          margin-bottom: 18px;
          align-items: start;
        }

        .board-card {
          padding-block: calc(18px * var(--board-panel-height-scale));
          padding-inline: 18px;
          border-color: color-mix(in srgb, var(--board-accent) 38%, rgba(255, 255, 255, 0.14));
        }

        .board-card h2 {
          font-size: clamp(1.28rem, 1.75vw, 1.6rem);
        }

        .board-card--interactive {
          box-shadow:
            var(--shadow),
            0 0 0 2px color-mix(in srgb, var(--board-accent) 55%, transparent);
        }

        .board-card--pulse {
          animation: board-pulse 280ms ease;
        }

        .board-card__header,
        .terminal-shell__header {
          display: flex;
          gap: 18px;
          align-items: flex-end;
          justify-content: space-between;
        }

        .board-card__meta {
          display: grid;
          gap: calc(8px * var(--board-panel-height-scale));
          min-width: 0;
        }

        .board-status {
          min-width: 150px;
          padding-block: calc(14px * var(--board-panel-height-scale));
          padding-inline: 16px;
          border-radius: 18px;
          background: rgba(255, 255, 255, 0.06);
        }

        .board-status__label,
        .stat-card__label,
        .inference-card__label,
        .payload-shell__label {
          display: block;
          font-size: 0.74rem;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          color: rgba(255, 233, 207, 0.6);
        }

        .board-status__value,
        .stat-card__value,
        .inference-card__value {
          display: block;
          margin-top: calc(8px * var(--board-panel-height-scale));
          font-size: 1rem;
        }

        .lane-tools {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 8px;
          margin-top: 0;
          padding-block: calc(6px * var(--board-panel-height-scale));
          padding-inline: 10px;
          border-radius: 14px;
          background: rgba(255, 255, 255, 0.06);
        }

        .lane-tools--agent {
          background: rgba(29, 143, 136, 0.12);
        }

        .keycap {
          display: inline-flex;
          width: calc(30px * var(--board-panel-height-scale));
          height: calc(30px * var(--board-panel-height-scale));
          align-items: center;
          justify-content: center;
          border-radius: 12px;
          background: rgba(255, 255, 255, 0.12);
          font-family: var(--font-display);
          font-size: 0.95rem;
          font-weight: 700;
        }

        .keycap--agent {
          background: rgba(29, 143, 136, 0.22);
        }

        .board-grid-shell {
          margin-top: calc(12px * var(--board-panel-height-scale));
          padding-block: calc(14px * var(--board-panel-height-scale));
          padding-inline: 14px;
          border-radius: 20px;
          background: var(--panel-soft);
        }

        .board-hud {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          column-gap: 12px;
          row-gap: calc(12px * var(--board-panel-height-scale));
          margin-bottom: calc(14px * var(--board-panel-height-scale));
        }

        .stat-card {
          padding-block: calc(14px * var(--board-panel-height-scale));
          padding-inline: 14px;
          border-radius: 18px;
          background: rgba(255, 255, 255, 0.08);
        }

        .board-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          column-gap: 12px;
          row-gap: calc(12px * var(--board-panel-height-scale));
          padding-block: calc(12px * var(--board-panel-height-scale));
          padding-inline: 12px;
          border-radius: 22px;
          background: rgba(255, 255, 255, 0.08);
        }

        .tile {
          display: grid;
          place-items: center;
          aspect-ratio: 1 / var(--board-panel-height-scale);
          border-radius: 20px;
          font-family: var(--font-display);
          font-size: clamp(1.2rem, 2vw, 2rem);
          font-weight: 800;
          letter-spacing: 0.02em;
          background: rgba(255, 250, 244, 0.12);
          color: #291911;
          transition: transform 160ms ease, background 160ms ease;
        }

        .tile[data-exp="0"] {
          color: transparent;
          background: rgba(255, 250, 244, 0.08);
        }

        .tile[data-exp="1"] { background: #f4d8b1; }
        .tile[data-exp="2"] { background: #efc186; }
        .tile[data-exp="3"] { background: #efaa65; }
        .tile[data-exp="4"] { background: #ef8f4e; color: #fff8ec; }
        .tile[data-exp="5"] { background: #e36d38; color: #fff8ec; }
        .tile[data-exp="6"] { background: #cc4f2d; color: #fff8ec; }
        .tile[data-exp="7"] { background: #ae3f31; color: #fff8ec; }
        .tile[data-exp="8"] { background: #8e6632; color: #fff8ec; }
        .tile[data-exp="9"] { background: #5f8f7e; color: #fff8ec; }
        .tile[data-exp="10"] { background: #2d7c76; color: #fff8ec; }
        .tile[data-exp="11"],
        .tile[data-exp="12"] {
          background: #195d6a;
          color: #fff8ec;
        }

        .tile--changed {
          animation: tile-pop 260ms ease;
        }

        .terminal-shell {
          padding: 18px;
          background: linear-gradient(180deg, rgba(29, 37, 48, 0.98), rgba(17, 22, 31, 0.98));
        }

        .terminal-shell__header {
          align-items: center;
        }

        .tab-nav {
          display: flex;
          gap: 8px;
          padding: 6px;
          border-radius: 18px;
          background: rgba(255, 255, 255, 0.06);
        }

        .tab-nav__button {
          padding: 10px 16px;
          border: 0;
          border-radius: 14px;
          background: transparent;
          color: rgba(239, 246, 255, 0.72);
          font: inherit;
          cursor: pointer;
        }

        .tab-nav__button.is-active {
          background: rgba(238, 165, 94, 0.18);
          color: #fff6eb;
        }

        .tab-panels {
          margin-top: 18px;
        }

        .terminal-window,
        .payload-shell {
          padding: 18px;
          border-radius: 22px;
          background: rgba(255, 255, 255, 0.06);
        }

        .terminal-window {
          display: grid;
          gap: 10px;
          align-content: start;
          height: var(--terminal-log-height);
          overflow-x: hidden;
          overflow-y: auto;
          overscroll-behavior: contain;
          font-family: var(--font-mono);
          scrollbar-color: rgba(255, 255, 255, 0.28) rgba(255, 255, 255, 0.06);
        }

        .terminal-window::-webkit-scrollbar {
          width: 10px;
        }

        .terminal-window::-webkit-scrollbar-track {
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.06);
        }

        .terminal-window::-webkit-scrollbar-thumb {
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.22);
        }

        .terminal-window::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.32);
        }

        .terminal-line {
          display: flex;
          gap: 12px;
          align-items: baseline;
          color: rgba(236, 243, 255, 0.88);
        }

        .terminal-line__channel {
          min-width: 72px;
          color: #74d1c8;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          font-size: 0.75rem;
        }

        .inference-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 12px;
        }

        .inference-card {
          padding: 16px;
          border-radius: 18px;
          background: rgba(255, 255, 255, 0.06);
        }

        .inference-live {
          display: grid;
          gap: 14px;
          margin-top: 14px;
        }

        .inference-metrics-grid,
        .inference-chart-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
        }

        .inference-card--chart {
          padding-bottom: 14px;
        }

        .inference-controls {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
        }

        .segmented-control {
          display: inline-flex;
          gap: 6px;
          background: rgba(255, 255, 255, 0.08);
          border-radius: 12px;
          padding: 4px;
        }

        .segmented-control__button {
          border: 0;
          background: transparent;
          color: rgba(236, 243, 255, 0.75);
          font: inherit;
          font-size: 0.82rem;
          padding: 6px 10px;
          border-radius: 9px;
          cursor: pointer;
        }

        .segmented-control__button.is-active {
          background: rgba(238, 165, 94, 0.22);
          color: #fff6eb;
        }

        .segmented-control__select {
          border: 0;
          background: rgba(255, 255, 255, 0.08);
          color: rgba(236, 243, 255, 0.85);
          font: inherit;
          font-size: 0.82rem;
          padding: 6px 10px;
          border-radius: 12px;
          cursor: pointer;
        }

        .inference-speed-row {
          display: flex;
          flex-direction: column;
          gap: 8px;
          margin-top: 12px;
        }

        .inference-speed-row__label {
          font-size: 0.84rem;
          color: rgba(236, 243, 255, 0.76);
        }

        .inference-speed-row input[type="range"] {
          width: 100%;
          min-width: 0;
          accent-color: rgba(238, 165, 94, 0.85);
        }

        .inference-note {
          margin: 8px 0 0;
          color: rgba(236, 243, 255, 0.72);
          font-size: 0.84rem;
        }

        .inference-chart {
          width: 100%;
          height: 160px;
          margin-top: 10px;
          border-radius: 12px;
          background: rgba(12, 16, 24, 0.46);
        }

        .action-distribution {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 10px;
          margin-top: 10px;
        }

        .action-bar {
          display: grid;
          gap: 8px;
          justify-items: center;
        }

        .action-bar__label {
          font-size: 0.78rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: rgba(236, 243, 255, 0.74);
        }

        .action-bar__track {
          width: 100%;
          height: 92px;
          border-radius: 10px;
          background: rgba(255, 255, 255, 0.08);
          display: flex;
          align-items: flex-end;
          padding: 4px;
        }

        .action-bar__fill {
          width: 100%;
          border-radius: 8px;
          background: linear-gradient(180deg, rgba(116, 209, 200, 0.95), rgba(29, 143, 136, 0.95));
          min-height: 2px;
        }

        .action-bar__value {
          font-family: var(--font-mono);
          font-size: 0.85rem;
          color: rgba(236, 243, 255, 0.88);
        }

        .payload-shell {
          margin-top: 14px;
        }

        .payload-shell__code {
          margin: 12px 0 0;
          white-space: pre-wrap;
          font-family: var(--font-mono);
          color: rgba(236, 243, 255, 0.9);
        }

        [hidden] {
          display: none !important;
        }

        @keyframes tile-pop {
          0% { transform: scale(1); }
          45% { transform: scale(1.06); }
          100% { transform: scale(1); }
        }

        @keyframes board-pulse {
          0% { box-shadow: var(--shadow); }
          50% { box-shadow: 0 18px 44px color-mix(in srgb, var(--board-accent) 35%, rgba(0, 0, 0, 0.18)); }
          100% { box-shadow: var(--shadow); }
        }

        @media (max-width: 1080px) {
          .board-layout,
          .inference-grid {
            grid-template-columns: 1fr;
          }

          .inference-metrics-grid,
          .inference-chart-grid,
          .action-distribution {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 720px) {
          .app-shell {
            padding: 20px 16px 28px;
          }

          .hero__panel,
          .board-card,
          .terminal-shell {
            border-radius: 24px;
          }

          .board-card__header,
          .terminal-shell__header {
            flex-direction: column;
          }

          .board-status,
          .tab-nav {
            width: 100%;
          }

          .board-hud {
            grid-template-columns: 1fr;
          }

          .tile {
            border-radius: 16px;
          }
        }
        """
    ).strip()
