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

  let ws = null;
  let sendInterval = null;
  let attempt = 0;

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
      } catch (e) {
        console.warn("Invalid message", e, ev.data);
      }
    });

    ws.addEventListener("close", () => {
      logStatus("disconnected");
      stopSendingFrames();
      scheduleReconnect();
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
    rightCard.innerHTML = '<div class="prediction"><div class="pred-label">' +
      String(prediction ?? "unknown") +
      '</div><div class="pred-conf">' + confStr + '</div></div>';
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

  // initialize
  (async () => {
    await startCamera();
    connectWebSocket();
  })();

})();