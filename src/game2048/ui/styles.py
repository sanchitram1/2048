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
          padding: 32px 24px 40px;
        }

        .hero {
          display: block;
          margin-bottom: 24px;
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
          padding: 30px;
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
          font-size: clamp(2.8rem, 5vw, 4.8rem);
          line-height: 0.95;
          max-width: none;
        }

        .hero__lede,
        .lane-tools__hint {
          margin: 12px 0 0;
          color: var(--text-muted);
          line-height: 1.5;
        }

        .board-layout {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 24px;
          margin-bottom: 24px;
        }

        .board-card {
          padding: 24px;
          border-color: color-mix(in srgb, var(--board-accent) 38%, rgba(255, 255, 255, 0.14));
        }

        .board-card--interactive {
          transform: translateY(-8px);
        }

        .board-card--pulse {
          animation: board-pulse 280ms ease;
        }

        .board-card__header,
        .terminal-shell__header {
          display: flex;
          gap: 18px;
          align-items: flex-start;
          justify-content: space-between;
        }

        .board-status {
          min-width: 150px;
          padding: 14px 16px;
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
          margin-top: 8px;
          font-size: 1rem;
        }

        .lane-tools {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 10px;
          margin-top: 18px;
          padding: 14px 16px;
          border-radius: 20px;
          background: rgba(255, 255, 255, 0.06);
        }

        .lane-tools--agent {
          background: rgba(29, 143, 136, 0.12);
        }

        .keycap {
          display: inline-flex;
          width: 38px;
          height: 38px;
          align-items: center;
          justify-content: center;
          border-radius: 14px;
          background: rgba(255, 255, 255, 0.12);
          font-family: var(--font-display);
          font-size: 1.1rem;
          font-weight: 700;
        }

        .keycap--agent {
          background: rgba(29, 143, 136, 0.22);
        }

        .board-grid-shell {
          margin-top: 18px;
          padding: 18px;
          border-radius: 24px;
          background: var(--panel-soft);
        }

        .board-hud {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 12px;
          margin-bottom: 14px;
        }

        .stat-card {
          padding: 14px;
          border-radius: 18px;
          background: rgba(255, 255, 255, 0.08);
        }

        .board-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 12px;
          padding: 12px;
          border-radius: 22px;
          background: rgba(255, 255, 255, 0.08);
        }

        .tile {
          display: grid;
          place-items: center;
          aspect-ratio: 1;
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
          padding: 24px;
          background: linear-gradient(180deg, rgba(29, 37, 48, 0.98), rgba(17, 22, 31, 0.98));
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
          min-height: 240px;
          font-family: var(--font-mono);
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
