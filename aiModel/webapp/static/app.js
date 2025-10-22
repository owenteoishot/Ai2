/*
Minimal frontend logic for AI Charades webapp.
- Captures webcam frames every 500ms
- Sends base64 JPEG frames over WebSocket to /ws
- Displays predictions returned by the backend

Behavior:
  Open the page served by the backend (e.g. http://127.0.0.1:8000/static/index.html).
*/
(() => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const overlay = document.getElementById("overlay");
  const status = document.getElementById("status");
  const rightCard = document.getElementById("rightCard"); // query right-side card safely (may be null)

  // Game UI elements (safe no-op if not present)
  const gameCurrent = document.getElementById("gameCurrent");
  const gameScore = document.getElementById("gameScore");
  const gameTimer = document.getElementById("gameTimer");
  const startBtn = document.getElementById("startBtn");
  const resetBtn = document.getElementById("resetBtn");
  // Defensive references: prefer direct ids, fallback to searching inside rightCard without modifying rightCard DOM
  const startBtnEl = startBtn || (rightCard && rightCard.querySelector("#startBtn"));
  const resetBtnEl = resetBtn || (rightCard && rightCard.querySelector("#resetBtn"));

  let ws = null;
  let sendInterval = null;
  let attempt = 0;

  // UI state for micro-interactions
  let lastScore = 0;
  let lastCurrentAction = null;


  // Visual bump animation for score
  function bumpScoreAnimation(el) {
    if (!el) return;
    el.classList.add('score-bump');
    setTimeout(() => el.classList.remove('score-bump'), 550);
  }


  function logStatus(txt) {
    status.textContent = txt;
  }

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      await video.play();
      // match canvas to video resolution
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
    } catch (e) {
      overlay.textContent = "Camera error: " + e.message;
      logStatus("camera_error");
      console.error(e);
    }
  }

  function makeWsUrl() {
    const proto = (location.protocol === "https:") ? "wss" : "ws";
    // connect to same host/port the page was served from
    return `${proto}://${location.host}/ws`;
  }

  function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    const url = makeWsUrl();
    logStatus("connecting...");
    ws = new WebSocket(url);

    ws.addEventListener("open", () => {
      attempt = 0;
      logStatus("connected");
      overlay.textContent = "Waiting for predictions...";
      startSendingFrames();
      setGameButtonsEnabled(true);
    });

    ws.addEventListener("message", (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.error) {
          overlay.textContent = `Error: ${msg.error}`;
          return;
        }
        // robust label/conf extraction supporting legacy and new message formats
        let label = msg.label ?? msg.prediction ?? "unknown"; // prefer label, fallback to prediction
        let conf = msg.confidence ?? msg.confidence_pct ?? 0; // support either confidence field
        // preserve existing overlay formatting/behavior for on-video text (do not change)
        const overlayConf = (typeof msg.confidence === "number") ? (msg.confidence * 100).toFixed(1) + "%" : "";
        overlay.textContent = `${label} ${overlayConf}`;
        // also update the separate right-side card (safe no-op if rightCard missing)
        updateRightCard(label, conf);
        // update game UI if backend included a game payload
        if (msg.game) {
          updateGameUI(msg.game);
        }
      } catch (e) {
        console.warn("Invalid message", e, ev.data);
      }
    });

    ws.addEventListener("close", () => {
      logStatus("disconnected");
      stopSendingFrames();
      scheduleReconnect();
      setGameButtonsEnabled(false);
    });

    ws.addEventListener("error", (e) => {
      console.warn("WebSocket error", e);
      ws.close();
    });
  }

  function scheduleReconnect() {
    attempt++;
    const backoff = Math.min(30, Math.pow(1.5, attempt)); // seconds
    logStatus(`reconnecting in ${Math.round(backoff)}s`);
    setTimeout(() => connectWebSocket(), backoff * 1000);
  }

  function startSendingFrames() {
    if (sendInterval) return;
    // capture every 100ms
    sendInterval = setInterval(captureAndSend, 100);
    captureAndSend(); // immediate
  }

  function stopSendingFrames() {
    if (sendInterval) {
      clearInterval(sendInterval);
      sendInterval = null;
    }
  }

  // updateRightCard(prediction, confidence): safely update the right-side card display (no-op if missing)
  function updateRightCard(prediction, confidence) {
    if (!rightCard) return; // leave rightCard untouched when not present
    // normalize confidence to a number and treat values <=1 as fractions
    const confNum = Number(confidence) || 0;
    const confPct = (confNum > 0 && confNum <= 1) ? confNum * 100 : confNum;
    const confStr = confPct.toFixed(1) + "%";
    // Update only #prediction to avoid removing game controls in rightCard
    const predEl = document.getElementById("prediction");
    const innerMarkup = '<div class="pred-label">' + String(prediction ?? "unknown") +
      '</div><div class="pred-conf">' + confStr + '</div>';
    if (predEl) {
      predEl.innerHTML = innerMarkup;
      return;
    }
    // Fallback: if #prediction element missing, append a non-destructive child so we don't overwrite other children.
    try {
      if (rightCard) {
        const newEl = document.createElement("div");
        newEl.id = "prediction";
        newEl.className = "prediction";
        newEl.innerHTML = innerMarkup;
        rightCard.appendChild(newEl);
      }
    } catch (e) {
      console.warn("updateRightCard fallback failed", e);
    }
  }

  // Update game UI elements when backend includes a game payload (safe no-op)
  function updateGameUI(game) {
    try {
      // Defensive reads
      const remaining = Number(game.remaining ?? 0);
      const isActive = Boolean(game.isGameActive);
      const curAction = String(game.currentAction ?? "—");
      const scoreVal = Number(game.score ?? 0);

      // Update timer and score (preserve existing behavior)
      if (gameScore) gameScore.textContent = String(scoreVal);
      if (gameTimer) gameTimer.textContent = String(game.remaining ?? 0);

      // Detect score increase -> bump animation for positive feedback
      if (scoreVal > lastScore) {
        if (gameScore) bumpScoreAnimation(gameScore);
      }
      lastScore = scoreVal;

      // If remaining is zero (backend controls remaining/isGameActive), show a clear game-over message,
      // add a "game-over" class to the current display. Do not disable Start on Game Over — allow immediate restart by user.
      if (remaining <= 0) {
        if (gameCurrent) {
          gameCurrent.textContent = "⛔ Game Over";
          gameCurrent.classList.add("game-over");
        }
        // Ensure Start remains enabled so user can restart; keep Reset enabled so the user can clear state.
        // Do not disable Start on Game Over — allow immediate restart by user.
        try {
          const sBtn = startBtn || (rightCard && rightCard.querySelector("#startBtn"));
          const rBtn = resetBtn || (rightCard && rightCard.querySelector("#resetBtn"));
          if (sBtn) sBtn.disabled = false;
          if (rBtn) rBtn.disabled = false;
        } catch (e) {
          // ignore button errors
        }
        lastCurrentAction = null; // reset last current so when user restarts it will animate
      } else {
        // Game active: show current action and remove game-over styling
        if (gameCurrent) {
          // animate change of current action
          if (lastCurrentAction !== curAction) {
            // small pulse to draw attention to new prompt
            gameCurrent.classList.add("pulse");
            setTimeout(() => gameCurrent.classList.remove("pulse"), 600);
          }
          gameCurrent.textContent = `🎯 Current: ${curAction}`;
          gameCurrent.classList.remove("game-over");
        }
        lastCurrentAction = curAction;
      }

      // visually indicate inactive state by toggling a class on rightCard (preserve original behavior)
      if (rightCard) {
        if (isActive) rightCard.classList.remove("game-inactive");
        else rightCard.classList.add("game-inactive");
      }
    } catch (e) {
      // defensive: do not break main flow
      console.warn("updateGameUI failed", e);
    }
  }

  // Enable/disable game buttons depending on WebSocket state
  function setGameButtonsEnabled(enabled) {
    try {
      // Try original references first; if the DOM was partially replaced earlier, query inside rightCard as a fallback.
      const sBtn = startBtn || (rightCard && rightCard.querySelector("#startBtn"));
      const rBtn = resetBtn || (rightCard && rightCard.querySelector("#resetBtn"));
      if (sBtn) sBtn.disabled = !enabled;
      if (rBtn) rBtn.disabled = !enabled;
    } catch (e) {
      console.warn("setGameButtonsEnabled failed", e);
    }
  }

  function captureAndSend() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!video.videoWidth || !video.videoHeight) return;

    // ensure canvas matches video size
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext("2d");
    // draw mirrored so sent frames match what user sees
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    // quality 0.8 JPEG for smaller payloads
    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
    const payload = JSON.stringify({ frame: dataUrl });
    try {
      ws.send(payload);
    } catch (e) {
      console.warn("Failed to send frame", e);
    }
  }

  // UI click to retry connection
  status.addEventListener("click", () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) connectWebSocket();
  });

  // wire up game buttons (safe no-op if buttons missing)
  if (startBtnEl) {
    startBtnEl.addEventListener("click", () => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      try { ws.send(JSON.stringify({ cmd: "game", action: "start" })); } catch (e) { console.warn("Failed to send start", e); }
    });
  }
  if (resetBtnEl) {
    resetBtnEl.addEventListener("click", () => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      try { ws.send(JSON.stringify({ cmd: "game", action: "reset" })); } catch (e) { console.warn("Failed to send reset", e); }
    });
  }
  // Fallback wiring: if the original button references were lost due to an earlier destructive overwrite,
  // attempt to find the buttons inside rightCard without modifying rightCard.innerHTML.
  if (!startBtnEl && rightCard) {
    const s = rightCard.querySelector("#startBtn");
    if (s) {
      s.addEventListener("click", () => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        try { ws.send(JSON.stringify({ cmd: "game", action: "start" })); } catch (e) { console.warn("Failed to send start (fallback)", e); }
      });
    }
  }
  if (!resetBtnEl && rightCard) {
    const r = rightCard.querySelector("#resetBtn");
    if (r) {
      r.addEventListener("click", () => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        try { ws.send(JSON.stringify({ cmd: "game", action: "reset" })); } catch (e) { console.warn("Failed to send reset (fallback)", e); }
      });
    }
  }

  // ensure buttons reflect initial WS state
  setGameButtonsEnabled(false);

  // initialize
  (async () => {
    await startCamera();
    connectWebSocket();
  })();

})();