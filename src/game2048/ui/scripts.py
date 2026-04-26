from __future__ import annotations

from textwrap import dedent


def render_scripts() -> str:
    return dedent(
        """
        const appStateNode = document.getElementById("mock-ui-state");
        const appState = appStateNode ? JSON.parse(appStateNode.textContent) : { boards: {} };
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

        const humanBoard = selectBoard("human-board");
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
            const ws = humanBoard.__humanWs;
            if (!ws || ws.readyState !== WebSocket.OPEN) {
              return;
            }
            const moveKey = moveWordToKey[moveWord];
            ws.send(JSON.stringify({ move: moveKey }));
          });

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
            updateBoard("human-board", 0, "Disconnected — refresh the page to reconnect");
            appendTerminalFromFullLine("[system] WebSocket /ws/human closed");
          });
        }
        """
    ).strip()
