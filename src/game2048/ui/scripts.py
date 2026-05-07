from __future__ import annotations

from textwrap import dedent


def render_scripts() -> str:
    return dedent(
        """
        const appStateNode = document.getElementById("mock-ui-state");
        const appState = appStateNode ? JSON.parse(appStateNode.textContent) : { boards: {} };
        const inferenceState = {
          episodePoints: [],
          completedEpisodes: 0,
          actionWindow: [],
          cumulativeActionCounts: { left: 0, right: 0, up: 0, down: 0 },
          actionMode: "rolling",
        };
        const keyToMove = {
          ArrowUp: "up",
          ArrowRight: "right",
          ArrowDown: "down",
          ArrowLeft: "left",
        };
        const moveWordToKey = { up: "u", right: "r", down: "d", left: "l" };

        function formatTile(exp) {
          return exp > 0 ? String(2 ** exp) : "";
        }

        function selectBoard(boardId) {
          return document.querySelector(`[data-board-id="${boardId}"]`);
        }

        function updateBoard(boardId, frameIndex, statusText) {
          const boardState = appState.boards[boardId];
          const boardElement = selectBoard(boardId);
          if (!boardState || !boardElement) {
            return;
          }

          const frame = boardState.frames[frameIndex % boardState.frames.length];
          const tiles = boardElement.querySelectorAll("[data-tile-index]");
          tiles.forEach((tile, index) => {
            const nextExp = frame.tiles[index] ?? 0;
            const previousExp = Number(tile.dataset.exp || "0");
            tile.dataset.exp = String(nextExp);
            tile.textContent = formatTile(nextExp);
            tile.classList.toggle("tile--changed", previousExp !== nextExp);
          });

          boardElement.querySelector('[data-stat-value="score"]').textContent = String(frame.score);
          boardElement.querySelector('[data-stat-value="best"]').textContent = String(frame.maxTile);
          boardElement.querySelector('[data-stat-value="moves"]').textContent = String(frame.moveCount);

          if (statusText) {
            boardElement.querySelector("[data-board-status]").textContent = statusText;
          }

          boardElement.classList.remove("board-card--pulse");
          void boardElement.offsetWidth;
          boardElement.classList.add("board-card--pulse");
        }

        function appendTerminalLine(channel, message) {
          const terminal = document.getElementById("terminal-log");
          if (!terminal) {
            return;
          }

          const line = document.createElement("div");
          line.className = "terminal-line";

          const channelNode = document.createElement("span");
          channelNode.className = "terminal-line__channel";
          channelNode.textContent = channel;

          const messageNode = document.createElement("span");
          messageNode.textContent = message;

          line.append(channelNode, messageNode);
          terminal.append(line);
          terminal.scrollTop = terminal.scrollHeight;
        }

        function appendTerminalFromFullLine(line) {
          const terminal = document.getElementById("terminal-log");
          if (!terminal) {
            return;
          }
          const match = /^\\[([^\\]]+)\\]\\s*(.*)$/.exec(line);
          if (match) {
            appendTerminalLine(match[1], match[2]);
            return;
          }
          const row = document.createElement("div");
          row.className = "terminal-line";
          row.textContent = line;
          terminal.append(row);
          terminal.scrollTop = terminal.scrollHeight;
        }

        function activateTab(tabName) {
          document.querySelectorAll("[data-tab-trigger]").forEach((button) => {
            button.classList.toggle("is-active", button.dataset.tabTrigger === tabName);
          });

          document.querySelectorAll("[data-tab-panel]").forEach((panel) => {
            const isActive = panel.dataset.tabPanel === tabName;
            panel.hidden = !isActive;
            panel.classList.toggle("is-active", isActive);
          });
        }

        document.querySelectorAll("[data-tab-trigger]").forEach((button) => {
          button.addEventListener("click", () => activateTab(button.dataset.tabTrigger));
        });

        function humanWsUrl() {
          const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
          return `${proto}//${window.location.host}/ws/human`;
        }

        function agentWsUrl(agentType) {
          const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
          const typeParam = agentType ? `?agent=${encodeURIComponent(agentType)}` : "";
          return `${proto}//${window.location.host}/ws/agent${typeParam}`;
        }

        function matchWsUrl(agentType, seedStr) {
          const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
          let q = `agent=${encodeURIComponent(agentType)}`;
          if (seedStr !== null && seedStr !== undefined && String(seedStr).trim() !== "") {
            q += `&seed=${encodeURIComponent(String(seedStr).trim())}`;
          }
          return `${proto}//${window.location.host}/ws/match?${q}`;
        }

        let gameMode = "play_against";
        let continueAgentActive = false;
        let continueAgentTickId = null;
        let sharedAgentStepMs = 700;

        function stopContinueAgentLoop() {
          continueAgentActive = false;
          if (continueAgentTickId) {
            window.clearTimeout(continueAgentTickId);
            continueAgentTickId = null;
          }
          const stepBtn = document.getElementById("inf-step-agent");
          if (stepBtn) {
            stepBtn.disabled = false;
            stepBtn.textContent = "Continue Agent";
          }
        }

        function requestContinueAgentStep() {
          if (!continueAgentActive || gameMode !== "play_against") {
            return;
          }
          const mws = humanBoard ? humanBoard.__matchWs : null;
          if (!mws || mws.readyState !== WebSocket.OPEN) {
            stopContinueAgentLoop();
            return;
          }
          const stepBtn = document.getElementById("inf-step-agent");
          if (stepBtn) {
            stepBtn.disabled = true;
            stepBtn.textContent = "Continuing...";
          }
          mws.send(JSON.stringify({ command: "step_agent" }));
        }

        function applyHumanFrameFromMatch(human) {
          const humanState = appState.boards["human-board"];
          if (!humanState || !humanState.frames.length) {
            return;
          }
          humanState.frames[0] = {
            tiles: human.tiles,
            score: human.score,
            moveCount: human.move_count,
            maxTile: human.max_tile,
            caption: "",
          };
          let statusText = "Match — your turn (arrow keys)";
          if (human.done) {
            statusText = "Game over (you)";
          }
          updateBoard("human-board", 0, statusText);
        }

        function applyAgentFrameFromMatch(agent) {
          const agentState = appState.boards["agent-board"];
          if (!agentState || !agentState.frames.length) {
            return;
          }
          agentState.frames[0] = {
            tiles: agent.tiles,
            score: agent.score,
            moveCount: agent.move_count,
            maxTile: agent.max_tile,
            caption: "",
          };
          const statusText = agent.done ? "Game over (agent)" : `Agent played ${agent.move || "…"}`;
          updateBoard("agent-board", 0, statusText);
        }

        function updateMatchSummary(match) {
          const box = document.getElementById("match-summary");
          const body = document.getElementById("match-summary-body");
          if (!box || !body) {
            return;
          }
          box.hidden = false;
          if (match.match_done && match.winner) {
            const who =
              match.winner === "human" ? "You win" : match.winner === "agent" ? "Agent wins" : "Tie";
            const first =
              match.first_finished === "human"
                ? "You reached game over first."
                : match.first_finished === "agent"
                  ? "Agent reached game over first."
                  : "";
            body.textContent = `${who}. ${match.win_reason || ""} ${first}`.trim();
            return;
          }
          let line = `${match.fairness_note || ""}`;
          if (match.human_done && !match.agent_done) {
            line += " You are done — click Continue Agent once.";
          } else if (!match.human_done && match.agent_done) {
            line += " Agent is done — keep playing your board.";
          }
          body.textContent = line;
        }

        function recordAgentMoveForCharts(agentMsg) {
          if (!agentMsg || !agentMsg.move) {
            return;
          }
          inferenceState.episodePoints.push({
            moveCount: Number(agentMsg.move_count || 0),
            score: Number(agentMsg.score || 0),
            maxTile: Number(agentMsg.max_tile || 0),
          });
          const moveName = String(agentMsg.move);
          if (moveName in inferenceState.cumulativeActionCounts) {
            inferenceState.cumulativeActionCounts[moveName] += 1;
          }
          inferenceState.actionWindow.push(moveName);
          if (inferenceState.actionWindow.length > 50) {
            inferenceState.actionWindow.shift();
          }
        }

        function applyMatchPayload(msg) {
          if (msg.event === "match_state") {
            inferenceState.episodePoints = [];
            inferenceState.actionWindow = [];
          }
          if (msg.event === "match_complete" && msg.match && msg.match.match_done) {
            inferenceState.completedEpisodes += 1;
          }
          if (msg.human) {
            applyHumanFrameFromMatch(msg.human);
          }
          if (msg.agent) {
            applyAgentFrameFromMatch(msg.agent);
            updateInferencePanelFromAgent(msg.agent);
            if (msg.last_agent_moved) {
              appendTerminalLine(
                "agent",
                `move=${msg.agent.move} score=${msg.agent.score} max=${msg.agent.max_tile} ${formatQValues(msg.agent.q_values)}`,
              );
              recordAgentMoveForCharts(msg.agent);
            }
            renderInferenceCharts();
          }
          if (msg.match) {
            updateMatchSummary(msg.match);
            const stepBtn = document.getElementById("inf-step-agent");
            if (stepBtn) {
              const showStep =
                gameMode === "play_against" &&
                msg.match.human_done &&
                !msg.match.agent_done &&
                !msg.match.match_done;
              stepBtn.hidden = !showStep;
              if (!showStep) {
                stopContinueAgentLoop();
              } else if (continueAgentActive) {
                if (continueAgentTickId) {
                  window.clearTimeout(continueAgentTickId);
                }
                continueAgentTickId = window.setTimeout(() => {
                  continueAgentTickId = null;
                  requestContinueAgentStep();
                }, sharedAgentStepMs);
              }
            }
          }
          if (msg.log_line) {
            appendTerminalFromFullLine(msg.log_line);
          }
          if (msg.event === "error" && msg.message) {
            appendTerminalFromFullLine(`[match] ${msg.message}`);
          }
          if (msg.event === "match_complete" && msg.match && msg.match.match_done) {
            const stepBtn = document.getElementById("inf-step-agent");
            if (stepBtn) {
              stepBtn.hidden = true;
            }
            stopContinueAgentLoop();
          }
          const payloadNode = document.getElementById("inference-payload-live");
          if (payloadNode) {
            payloadNode.textContent = JSON.stringify(msg, null, 2);
          }
        }

        function syncModeUI() {
          const isPlay = gameMode === "play_against";
          document.getElementById("mode-play-against")?.classList.toggle("is-active", isPlay);
          document.getElementById("mode-agent-autoplay")?.classList.toggle("is-active", !isPlay);
          const badge = document.getElementById("fairness-badge");
          if (badge) {
            badge.classList.toggle("fairness-badge--off", !isPlay);
          }
          const modeHint = document.getElementById("hint-play-against");
          if (modeHint) {
            modeHint.innerHTML = isPlay
              ? "Seed-matched start; trajectories diverge after your moves. If you reach game over first, click <strong>Continue Agent</strong> once to let the agent finish."
              : "Human and agent boards run independently in Agent Autoplay. Use the step delay slider to control autoplay speed.";
          }
          const foot = document.getElementById("game-controls-footnote");
          if (foot) {
            foot.innerHTML = isPlay
              ? "In <strong>Play Against Agent</strong>, press <strong>Start</strong> to open a match; <strong>Reset</strong> starts a new match."
              : "In <strong>Agent Autoplay</strong>, the agent runs on a timer. Use <strong>Stop</strong> to pause stepping and <strong>Reset</strong> for a new episode.";
          }
          if (!isPlay) {
            stopContinueAgentLoop();
            document.getElementById("inf-step-agent") && (document.getElementById("inf-step-agent").hidden = true);
            const ms = document.getElementById("match-summary");
            if (ms) {
              ms.hidden = true;
            }
          }
        }

        function applyHumanServerPayload(msg) {
          const humanState = appState.boards["human-board"];
          if (!humanState || !humanState.frames.length) {
            return;
          }
          const frame = {
            tiles: msg.tiles,
            score: msg.score,
            moveCount: msg.move_count,
            maxTile: msg.max_tile,
            caption: "",
          };
          humanState.frames[0] = frame;
          const statusText = msg.done ? "Game over" : "Focused — arrow keys send moves";
          updateBoard("human-board", 0, statusText);
          if (msg.log_line) {
            appendTerminalFromFullLine(msg.log_line);
          }
        }

        function formatQValues(qValues) {
          if (!Array.isArray(qValues)) {
            return "";
          }
          const labels = ["left", "right", "up", "down"];
          return qValues
            .map((value, index) => `${labels[index]}=${Number(value).toFixed(2)}`)
            .join(" ");
        }

        function setNodeText(id, value) {
          const node = document.getElementById(id);
          if (node) {
            node.textContent = String(value);
          }
        }

        function clampPercent(value) {
          if (!Number.isFinite(value)) {
            return 0;
          }
          return Math.max(0, Math.min(100, value));
        }

        function renderActionDistribution() {
          const host = document.getElementById("inf-action-distribution");
          if (!host) {
            return;
          }
          const actions = ["left", "right", "up", "down"];
          const counts = Object.fromEntries(actions.map((action) => [action, 0]));
          if (inferenceState.actionMode === "cumulative") {
            for (const action of actions) {
              counts[action] = Number(inferenceState.cumulativeActionCounts[action] || 0);
            }
          } else {
            for (const action of inferenceState.actionWindow) {
              if (action in counts) {
                counts[action] += 1;
              }
            }
          }
          const maxCount = Math.max(1, ...Object.values(counts));
          host.innerHTML = actions
            .map((action) => {
              const count = counts[action];
              const heightPct = clampPercent((count / maxCount) * 100);
              return `
                <div class="action-bar">
                  <span class="action-bar__label">${action}</span>
                  <div class="action-bar__track">
                    <div class="action-bar__fill" style="height:${heightPct}%"></div>
                  </div>
                  <span class="action-bar__value">${count}</span>
                </div>
              `;
            })
            .join("");
          const sampleNode = document.getElementById("inf-action-sample");
          const noteNode = document.getElementById("inf-action-mode-note");
          if (sampleNode) {
            const totalCumulative = actions
              .map((action) => Number(inferenceState.cumulativeActionCounts[action] || 0))
              .reduce((sum, value) => sum + value, 0);
            const sampleSize = inferenceState.actionMode === "cumulative"
              ? totalCumulative
              : inferenceState.actionWindow.length;
            sampleNode.textContent = `(N=${sampleSize})`;
          }
          if (noteNode) {
            noteNode.textContent = inferenceState.actionMode === "cumulative"
              ? "Showing cumulative actions for this page session."
              : "Showing last 50 moves.";
          }
        }

        function setActionMode(mode) {
          inferenceState.actionMode = mode === "cumulative" ? "cumulative" : "rolling";
          const rollingButton = document.getElementById("inf-mode-rolling");
          const cumulativeButton = document.getElementById("inf-mode-cumulative");
          if (rollingButton && cumulativeButton) {
            const isRolling = inferenceState.actionMode === "rolling";
            rollingButton.classList.toggle("is-active", isRolling);
            cumulativeButton.classList.toggle("is-active", !isRolling);
          }
          renderActionDistribution();
        }

        function renderLineChart(svgId, points, yKey, color, thresholds) {
          const svg = document.getElementById(svgId);
          if (!svg) {
            return;
          }
          const width = 360;
          const height = 140;
          const padding = { top: 10, right: 12, bottom: 24, left: 38 };
          const innerWidth = width - padding.left - padding.right;
          const innerHeight = height - padding.top - padding.bottom;

          if (!points.length) {
            svg.innerHTML = "";
            return;
          }

          function niceAxisMax(value) {
            const safe = Math.max(1, value);
            const magnitude = 10 ** Math.floor(Math.log10(safe));
            const normalized = safe / magnitude;
            let nice = 1;
            if (normalized <= 1) {
              nice = 1;
            } else if (normalized <= 2) {
              nice = 2;
            } else if (normalized <= 5) {
              nice = 5;
            } else {
              nice = 10;
            }
            return nice * magnitude;
          }

          const xMax = Math.max(1, points[points.length - 1].moveCount);
          const seriesYMax = Math.max(1, ...points.map((point) => Number(point[yKey] || 0)));
          const thresholdMax = Array.isArray(thresholds) && thresholds.length
            ? Math.max(...thresholds)
            : 0;
          const rawYMax = Math.max(seriesYMax, Math.min(thresholdMax, seriesYMax * 1.15));
          const yMax = niceAxisMax(rawYMax);

          function toX(moveCount) {
            return padding.left + (moveCount / xMax) * innerWidth;
          }

          function toY(value) {
            return padding.top + (1 - value / yMax) * innerHeight;
          }

          const pathData = points
            .map((point, index) => `${index === 0 ? "M" : "L"} ${toX(point.moveCount).toFixed(2)} ${toY(Number(point[yKey] || 0)).toFixed(2)}`)
            .join(" ");

          const thresholdLines = Array.isArray(thresholds)
            ? thresholds
                .filter((value) => value <= yMax)
                .map((value) => {
                  const y = toY(value).toFixed(2);
                  return `
                    <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="rgba(225,232,246,0.24)" stroke-dasharray="4 4" />
                    <text x="${width - padding.right - 2}" y="${(Number(y) - 2).toFixed(2)}" text-anchor="end" fill="rgba(225,232,246,0.62)" font-size="9">${value}</text>
                  `;
                })
                .join("")
            : "";

          const xTickValues = [0, Math.round(xMax / 2), xMax];
          const xTickLines = xTickValues
            .map((value) => {
              const x = toX(value).toFixed(2);
              return `
                <line x1="${x}" y1="${height - padding.bottom}" x2="${x}" y2="${height - padding.bottom + 4}" stroke="rgba(225,232,246,0.48)" />
                <text x="${x}" y="${height - 5}" text-anchor="middle" fill="rgba(225,232,246,0.72)" font-size="10">${value}</text>
              `;
            })
            .join("");

          const yTickValues = [0, Math.round(yMax / 2), yMax];
          const yTickLines = yTickValues
            .map((value) => {
              const y = toY(value).toFixed(2);
              return `
                <line x1="${padding.left - 4}" y1="${y}" x2="${padding.left}" y2="${y}" stroke="rgba(225,232,246,0.48)" />
                <text x="${padding.left - 8}" y="${(Number(y) + 3).toFixed(2)}" text-anchor="end" fill="rgba(225,232,246,0.72)" font-size="10">${value}</text>
              `;
            })
            .join("");

          svg.innerHTML = `
            <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="rgba(225,232,246,0.3)" />
            <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}" stroke="rgba(225,232,246,0.3)" />
            ${xTickLines}
            ${yTickLines}
            ${thresholdLines}
            <path d="${pathData}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
          `;
        }

        function renderInferenceCharts() {
          renderLineChart("inf-score-chart", inferenceState.episodePoints, "score", "#cf6b2d");
          renderLineChart(
            "inf-max-tile-chart",
            inferenceState.episodePoints,
            "maxTile",
            "#1d8f88",
            [64, 128, 256, 512, 1024],
          );
          renderActionDistribution();
        }

        function updateInferencePanelFromAgent(msg) {
          setNodeText("inf-current-score", msg.score ?? 0);
          setNodeText("inf-current-max-tile", msg.max_tile ?? 0);
          setNodeText("inf-current-moves", msg.move_count ?? 0);
          setNodeText("inf-episodes-complete", inferenceState.completedEpisodes);

          const payloadNode = document.getElementById("inference-payload-live");
          if (payloadNode) {
            payloadNode.textContent = JSON.stringify(msg, null, 2);
          }
        }

        function applyAgentServerPayload(msg) {
          const agentState = appState.boards["agent-board"];
          if (!agentState || !agentState.frames.length) {
            return;
          }
          const frame = {
            tiles: msg.tiles,
            score: msg.score,
            moveCount: msg.move_count,
            maxTile: msg.max_tile,
            caption: "",
          };
          agentState.frames[0] = frame;
          const statusText = msg.done ? "Game over" : `Model chose ${msg.move || "ready"}`;
          updateBoard("agent-board", 0, statusText);

          if (msg.event === "agent_move" || msg.event === "game_over") {
            appendTerminalLine(
              "agent",
              `move=${msg.move} score=${msg.score} max=${msg.max_tile} ${formatQValues(msg.q_values)}`,
            );
          }

          if (msg.event === "state") {
            inferenceState.episodePoints = [];
            inferenceState.actionWindow = [];
          }

          if (msg.event === "agent_move" || msg.event === "game_over") {
            inferenceState.episodePoints.push({
              moveCount: Number(msg.move_count || 0),
              score: Number(msg.score || 0),
              maxTile: Number(msg.max_tile || 0),
            });
            if (msg.move) {
              const moveName = String(msg.move);
              if (moveName in inferenceState.cumulativeActionCounts) {
                inferenceState.cumulativeActionCounts[moveName] += 1;
              }
              inferenceState.actionWindow.push(String(msg.move));
              if (inferenceState.actionWindow.length > 50) {
                inferenceState.actionWindow.shift();
              }
            }
          }
          if (msg.event === "game_over") {
            inferenceState.completedEpisodes += 1;
          }
          updateInferencePanelFromAgent(msg);
          renderInferenceCharts();
        }

        const humanBoard = selectBoard("human-board");
        const agentBoard = selectBoard("agent-board");

        function disconnectHumanAutoplay() {
          if (humanBoard && humanBoard.__humanWs) {
            humanBoard.__humanWs.close();
            humanBoard.__humanWs = null;
          }
        }

        function connectHumanAutoplay() {
          if (!humanBoard || gameMode !== "agent_autoplay") {
            return;
          }
          if (humanBoard.__humanWs && humanBoard.__humanWs.readyState === WebSocket.OPEN) {
            return;
          }
          const ws = new WebSocket(humanWsUrl());
          humanBoard.__humanWs = ws;
          ws.addEventListener("open", () => {
            updateBoard(
              "human-board",
              0,
              "Connected — click here if needed, then use arrow keys",
            );
          });
          ws.addEventListener("message", (event) => {
            const msg = JSON.parse(event.data);
            if (msg.event === "error") {
              appendTerminalFromFullLine(`[system] ${msg.message || "error"}`);
              return;
            }
            if (msg.event === "state" || msg.event === "move_result") {
              applyHumanServerPayload(msg);
            }
          });
          ws.addEventListener("close", () => {
            if (gameMode !== "agent_autoplay") {
              return;
            }
            updateBoard("human-board", 0, "Disconnected — refresh the page to reconnect");
            appendTerminalFromFullLine("[system] WebSocket /ws/human closed");
          });
        }

        if (humanBoard) {
          humanBoard.addEventListener("click", () => {
            humanBoard.focus();
          });

          humanBoard.addEventListener("keydown", (event) => {
            const moveWord = keyToMove[event.key];
            if (!moveWord) {
              return;
            }
            event.preventDefault();
            const moveKey = moveWordToKey[moveWord];
            if (gameMode === "play_against") {
              const mws = humanBoard.__matchWs;
              if (!mws || mws.readyState !== WebSocket.OPEN) {
                return;
              }
              mws.send(JSON.stringify({ command: "human_move", move: moveKey }));
              return;
            }
            const hws = humanBoard.__humanWs;
            if (!hws || hws.readyState !== WebSocket.OPEN) {
              return;
            }
            hws.send(JSON.stringify({ move: moveKey }));
          });
        }

        function closeMatchWebSocket() {
          stopContinueAgentLoop();
          if (humanBoard && humanBoard.__matchWs) {
            humanBoard.__matchWs.close();
            humanBoard.__matchWs = null;
          }
        }

        function connectMatchWebSocket() {
          if (!humanBoard) {
            return;
          }
          closeMatchWebSocket();
          const agentSelectEl = document.getElementById("inf-agent-select");
          const agentType = agentSelectEl ? agentSelectEl.value : "dqn";
          const mws = new WebSocket(matchWsUrl(agentType, null));
          humanBoard.__matchWs = mws;
          mws.addEventListener("open", () => {
            updateBoard("human-board", 0, "Match connected — your turn (arrow keys)");
            updateBoard("agent-board", 0, "Match — agent will move after you");
          });
          mws.addEventListener("message", (event) => {
            const msg = JSON.parse(event.data);
            if (msg.event === "model_missing") {
              updateBoard("agent-board", 0, msg.message || "no checkpoint found in models...");
              appendTerminalFromFullLine(`[match] ${msg.message || "model missing"}`);
              closeMatchWebSocket();
              return;
            }
            applyMatchPayload(msg);
          });
          mws.addEventListener("close", () => {
            stopContinueAgentLoop();
            humanBoard.__matchWs = null;
          });
        }

        if (agentBoard) {
          const rollingButton = document.getElementById("inf-mode-rolling");
          const cumulativeButton = document.getElementById("inf-mode-cumulative");
          if (rollingButton) {
            rollingButton.addEventListener("click", () => setActionMode("rolling"));
          }
          if (cumulativeButton) {
            cumulativeButton.addEventListener("click", () => setActionMode("cumulative"));
          }
          let agentStepTimeoutId = null;
          let autoplayWs = null;
          let autoplayStarted = false;
          let agentPlaybackPaused = false;
          let agentStepMs = 700;

          function cancelAgentCadence() {
            if (agentStepTimeoutId) {
              window.clearTimeout(agentStepTimeoutId);
              agentStepTimeoutId = null;
            }
          }

          function scheduleDelayedAgentSend() {
            if (!autoplayWs || autoplayWs.readyState !== WebSocket.OPEN || agentPlaybackPaused) {
              return;
            }
            cancelAgentCadence();
            agentStepTimeoutId = window.setTimeout(() => {
              agentStepTimeoutId = null;
              if (autoplayWs && autoplayWs.readyState === WebSocket.OPEN && !agentPlaybackPaused) {
                autoplayWs.send(JSON.stringify({ command: "step" }));
              }
            }, agentStepMs);
          }

          function connectAgentWs(agentType) {
            cancelAgentCadence();
            agentPlaybackPaused = false;
            autoplayStarted = true;
            if (autoplayWs && autoplayWs.readyState === WebSocket.OPEN) {
              autoplayWs.close();
            }
            autoplayWs = new WebSocket(agentWsUrl(agentType));
            agentBoard.__agentWs = autoplayWs;

            autoplayWs.addEventListener("open", () => {
              const label = agentType ? `Autoplay — ${agentType}` : "Autoplay connected";
              updateBoard("agent-board", 0, label);
            });

            autoplayWs.addEventListener("message", (event) => {
              const msg = JSON.parse(event.data);
              if (msg.event === "model_missing" || msg.event === "error") {
                updateBoard("agent-board", 0, msg.message || "Agent stream unavailable");
                appendTerminalFromFullLine(`[agent] ${msg.message || "agent stream error"}`);
                cancelAgentCadence();
                return;
              }
              if (msg.event === "state" || msg.event === "agent_move" || msg.event === "game_over") {
                if (msg.event === "game_over" || msg.done) {
                  cancelAgentCadence();
                }
                applyAgentServerPayload(msg);
                if (!msg.done) {
                  scheduleDelayedAgentSend();
                }
              }
            });

            autoplayWs.addEventListener("close", () => {
              cancelAgentCadence();
            });
          }

          const speedSlider = document.getElementById("inf-agent-speed");
          const speedValueEl = document.getElementById("inf-agent-speed-value");
          if (speedSlider && speedValueEl) {
            function syncAgentSpeedFromSlider() {
              agentStepMs = Number(speedSlider.value);
              sharedAgentStepMs = agentStepMs;
              speedValueEl.textContent = String(agentStepMs);
              speedSlider.setAttribute("aria-valuenow", speedSlider.value);
              if (agentStepTimeoutId) {
                cancelAgentCadence();
                scheduleDelayedAgentSend();
              }
            }
            speedSlider.addEventListener("input", syncAgentSpeedFromSlider);
            syncAgentSpeedFromSlider();
          }

          const agentSelect = document.getElementById("inf-agent-select");
          const agentStart = document.getElementById("inf-agent-start");
          const agentStop = document.getElementById("inf-agent-stop");
          const agentReset = document.getElementById("inf-agent-reset");
          const stepAgentBtn = document.getElementById("inf-step-agent");

          document.querySelectorAll("[data-game-mode]").forEach((btn) => {
            btn.addEventListener("click", () => {
              const mode = btn.dataset.gameMode;
              if (!mode || mode === gameMode) {
                return;
              }
              gameMode = mode;
              closeMatchWebSocket();
              cancelAgentCadence();
              if (autoplayWs && autoplayWs.readyState === WebSocket.OPEN) {
                autoplayWs.close();
              }
              autoplayWs = null;
              autoplayStarted = false;
              agentPlaybackPaused = false;
              if (gameMode === "play_against") {
                disconnectHumanAutoplay();
                const ms = document.getElementById("match-summary");
                if (ms) {
                  ms.hidden = true;
                }
                updateBoard("human-board", 0, "Select Play Against Agent, then press Start");
                updateBoard(
                  "agent-board",
                  0,
                  `Ready — ${agentSelect ? agentSelect.value : "dqn"} (press Start for match)`,
                );
              } else {
                connectHumanAutoplay();
                const ms = document.getElementById("match-summary");
                if (ms) {
                  ms.hidden = true;
                }
                updateBoard(
                  "agent-board",
                  0,
                  `Ready — ${agentSelect ? agentSelect.value : "dqn"} (press Start)`,
                );
              }
              syncModeUI();
            });
          });

          if (stepAgentBtn) {
            stepAgentBtn.addEventListener("click", () => {
              if (gameMode !== "play_against") {
                return;
              }
              if (continueAgentActive) {
                return;
              }
              continueAgentActive = true;
              requestContinueAgentStep();
            });
          }

          if (agentSelect && agentStart) {
            agentSelect.addEventListener("change", () => {
              if (gameMode === "play_against") {
                updateBoard(
                  "agent-board",
                  0,
                  `Ready — ${agentSelect.value} (press Start for new match)`,
                );
                return;
              }
              if (!autoplayStarted) {
                updateBoard("agent-board", 0, `Ready — ${agentSelect.value} (press Start)`);
                return;
              }
              connectAgentWs(agentSelect.value);
            });
          }
          if (agentSelect && agentStart) {
            agentStart.addEventListener("click", () => {
              if (gameMode === "play_against") {
                connectMatchWebSocket();
                return;
              }
              if (autoplayWs && autoplayWs.readyState === WebSocket.OPEN && agentPlaybackPaused) {
                agentPlaybackPaused = false;
                scheduleDelayedAgentSend();
                updateBoard("agent-board", 0, `Running — ${agentSelect.value}`);
                return;
              }
              connectAgentWs(agentSelect.value);
            });
          }
          if (agentStop) {
            agentStop.addEventListener("click", () => {
              if (gameMode === "play_against") {
                closeMatchWebSocket();
                updateBoard("human-board", 0, "Match stopped — press Start to reconnect");
                updateBoard("agent-board", 0, "Match stopped");
                return;
              }
              agentPlaybackPaused = true;
              cancelAgentCadence();
              if (autoplayWs && autoplayWs.readyState === WebSocket.OPEN) {
                updateBoard("agent-board", 0, "Paused — press Start to resume");
              }
            });
          }
          if (agentReset) {
            agentReset.addEventListener("click", () => {
              if (gameMode === "play_against") {
                const mws = humanBoard ? humanBoard.__matchWs : null;
                const agentType = agentSelect ? agentSelect.value : "dqn";
                const payload = { command: "reset", agent: agentType };
                if (mws && mws.readyState === WebSocket.OPEN) {
                  mws.send(JSON.stringify(payload));
                } else {
                  connectMatchWebSocket();
                }
                return;
              }
              if (autoplayWs && autoplayWs.readyState === WebSocket.OPEN) {
                autoplayWs.send(JSON.stringify({ command: "reset" }));
              }
            });
          }

          const selectedType = agentSelect ? agentSelect.value : "dqn";
          updateBoard("agent-board", 0, `Ready — ${selectedType} (press Start)`);
        }

        syncModeUI();
        if (gameMode === "play_against") {
          disconnectHumanAutoplay();
          updateBoard("human-board", 0, "Select Play Against Agent, then press Start");
          updateBoard("agent-board", 0, "Ready — dqn (press Start for match)");
        } else {
          connectHumanAutoplay();
        }
        """
    ).strip()
