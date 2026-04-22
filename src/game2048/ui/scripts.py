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
          boardElement.querySelector("[data-board-caption]").textContent = frame.caption;

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

        let humanFrameIndex = 0;
        let agentFrameIndex = 0;

        document.addEventListener("keydown", (event) => {
          const move = keyToMove[event.key];
          if (!move) {
            return;
          }

          event.preventDefault();
          const humanState = appState.boards["human-board"];
          const agentState = appState.boards["agent-board"];
          if (!humanState || !agentState) {
            return;
          }

          humanFrameIndex = (humanFrameIndex + 1) % humanState.frames.length;
          updateBoard("human-board", humanFrameIndex, `Last move: ${move}`);
          appendTerminalLine("human", `mocked ${move} move rendered in the left lane`);

          window.setTimeout(() => {
            agentFrameIndex = (agentFrameIndex + 1) % agentState.frames.length;
            updateBoard("agent-board", agentFrameIndex, "Mock response ready");
            appendTerminalLine(
              "agent",
              `placeholder inference advanced the right lane after human ${move}`,
            );
          }, 260);
        });
        """
    ).strip()
