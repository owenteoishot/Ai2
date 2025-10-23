#!/usr/bin/env python3
"""
aiModel/webapp/backend.py

FastAPI WebSocket backend for low-latency webcam classification.

Run instructions (example):
  python -m venv .venv
  .venv\Scripts\activate     # Windows
  pip install -r aiModel/requirements.txt
  python aiModel/webapp/backend.py --host 127.0.0.1 --port 8000 --model aiModel/models/ai_charades_tcn2.pt

Or use uvicorn directly:
  uvicorn aiModel.webapp.backend:app --host 127.0.0.1 --port 8000 --reload

Open in browser:
  http://127.0.0.1:8000/static/index.html

Optional optimizations: export model to ONNX or install CUDA-enabled PyTorch for GPU. To enable GPU inference install torch with CUDA and load the checkpoint with map_location='cuda'.

"""
import argparse
import asyncio
import base64
import json
import logging
import sys
import os
import time
from typing import List

import cv2
import mediapipe as mp
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import random

# Game constants
GAME_CONF_THRESH = 0.85
FALLBACK_ACTIONS = ["Running", "Nodding", "Calling", "Playing games", "Boxing", "Air guitar"]

# Try to reuse normalization and TCN from existing script; fallback copies are small.
try:
    from aiModel.run_webcam import normalize_keypoints, TCN  # type: ignore
except Exception:
    # Fallback: small local copy of normalize_keypoints. Remove this if importable.
    def normalize_keypoints(pts: np.ndarray) -> np.ndarray:
        """
        Fallback normalization logic copied from run_webcam.py.
        Centers at hips midpoint and scales by shoulder distance.
        Returns first two coordinates for each landmark.
        """
        left_hip, right_hip = pts[23], pts[24]
        center = (left_hip + right_hip) / 2
        pts = pts.copy()
        pts -= center
        left_shoulder, right_shoulder = pts[11], pts[12]
        scale = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6
        pts /= scale
        return pts[:, :2]

    # Fallback minimal TCN to allow loading state_dict if needed.
    import torch.nn as nn

    class TCN(nn.Module):
        def __init__(self, in_ch, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, 128, 5, padding=2),
                nn.ReLU(),
                nn.Conv1d(128, 128, 5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Linear(128, n_classes)

        def forward(self, x):
            z = self.net(x).squeeze(-1)
            return self.fc(z)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aiModel.webapp")

app = FastAPI(title="AI Charades Webapp")

# Mount static files (index.html + app.js expected)
app.mount("/static", StaticFiles(directory="aiModel/webapp/static"), name="static")


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/")
async def index():
    # serve the static index directly
    return FileResponse("aiModel/webapp/static/index.html")


async def run_inference(model: torch.nn.Module, actions: List[str], x_tensor: torch.Tensor):
    """
    Run model inference in a thread to avoid blocking the event loop.
    Returns (label, confidence)
    """
    def _infer():
        model.eval()
        with torch.no_grad():
            out = model(x_tensor)
            probs = torch.softmax(out, dim=1)
            action_id = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, action_id].item())
            label = actions[action_id] if actions is not None else str(action_id)
            return label, confidence

    return await asyncio.to_thread(_infer)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket handler:
    - receives JSON messages with {"frame": "<base64jpeg>"}
    - decodes to BGR, runs mediapipe pose, normalizes landmarks, buffers sequence,
      and when buffer == seq_len runs inference and returns {"label","confidence","timestamp"}.
    """
    await websocket.accept()
    seq_len = getattr(app.state, "seq_len", 48)
    model = getattr(app.state, "model", None)
    actions = getattr(app.state, "actions", None)
    
    # per-connection sliding buffer (in-memory)
    sequence: List[np.ndarray] = []
    
    # per-connection game state
    # Use explicit fallback: prefer provided non-empty actions list, otherwise FALLBACK_ACTIONS
    game = {"actions": (actions if actions is not None and len(actions) > 0 else FALLBACK_ACTIONS), "current": None, "prev": None, "score": 0, "end_ts": 0, "isActive": False, "duration": 30}

    def pick_next_action(g):
        """Pick a random action different from the previous one (if possible)."""
        acts = g.get("actions") or FALLBACK_ACTIONS
        # defensive: ensure acts is a list-like
        try:
            acts_list = list(acts)
        except Exception:
            # invalid actions list; clear current and prev, return None
            g["prev"] = g.get("current")
            g["current"] = None
            return None
        if not acts_list:
            g["prev"] = g.get("current")
            g["current"] = None
            return None

        prev = g.get("prev")
        # build candidates excluding prev
        candidates = [a for a in acts_list if a != prev]
        if candidates:
            cand = random.choice(candidates)
        else:
            # only candidate is same as prev (single-item list) — return it
            cand = acts_list[0]

        # update state: prev becomes previous current, current becomes chosen
        g["prev"] = g.get("current")
        g["current"] = cand
        return cand

    def game_payload(g):
        """Return a serializable game payload for the client."""
        now_ms = int(time.time() * 1000)
        remaining = max(0, int((g.get("end_ts", 0) - now_ms) / 1000))
        # auto-deactivate if time expired
        if g.get("isActive") and remaining <= 0:
            g["isActive"] = False
        return {"currentAction": g.get("current"), "score": int(g.get("score", 0)), "remaining": remaining, "isGameActive": bool(g.get("isActive", False))}

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    try:
        while True:
            data_text = await websocket.receive_text()
            try:
                pkt = json.loads(data_text)
            except Exception:
                await websocket.send_text(json.dumps({"error": "invalid_json"}))
                continue

            # handle control commands (game) sent over the same websocket
            if isinstance(pkt, dict) and pkt.get("cmd") == "game":
                action_cmd = pkt.get("action")
                now_ms = int(time.time() * 1000)
                if action_cmd == "start":
                    game["score"] = 0
                    game["isActive"] = True
                    game["end_ts"] = now_ms + int(game.get("duration", 30)) * 1000
                    pick_next_action(game)
                elif action_cmd == "reset":
                    game["isActive"] = False
                    game["score"] = 0
                    game["end_ts"] = 0
                    game["prev"] = None
                    game["current"] = None
                # send acknowledgement including current game state
                ack = {"cmd": "game", "action": action_cmd, "game": game_payload(game), "timestamp": now_ms}
                await websocket.send_text(json.dumps(ack))
                continue

            frame_b64 = pkt.get("frame")
            if not frame_b64:
                await websocket.send_text(json.dumps({"error": "no_frame"}))
                continue

            try:
                frame_bytes = base64.b64decode(frame_b64.split(",", 1)[-1])
                arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
                if img is None:
                    raise ValueError("cv2.imdecode returned None")
            except Exception as e:
                await websocket.send_text(json.dumps({"error": "decode_failed", "detail": str(e)}))
                continue

            # Mediapipe expects RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if not result.pose_landmarks:
                await websocket.send_text(json.dumps({"error": "no_pose"}))
                continue

            lm = result.pose_landmarks.landmark
            if len(lm) != 33:
                await websocket.send_text(json.dumps({"error": "no_pose"}))
                continue

            pts = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
            try:
                pts_norm = normalize_keypoints(pts)  # expect (33,2)
            except Exception as e:
                await websocket.send_text(json.dumps({"error": "normalize_failed", "detail": str(e)}))
                continue

            sequence.append(pts_norm.flatten())  # shape (features, )

            # keep sliding window
            if len(sequence) > seq_len:
                sequence = sequence[-seq_len:]

            if len(sequence) == seq_len:
                # prepare tensor: (1, features, seq_len)
                x = np.array(sequence, dtype=np.float32)
                x_tensor = torch.tensor(x).unsqueeze(0).permute(0, 2, 1)  # (1, features, seq_len)
                # run inference off the event loop
                try:
                    if model is None:
                        # safe fallback: return argmax of zeros if model missing
                        await websocket.send_text(json.dumps({"error": "model_not_loaded"}))
                    else:
                        label, confidence = await run_inference(model, actions, x_tensor)
                        # game evaluation: check match and update score/target when active
                        now_ms = int(time.time() * 1000)
                        remaining = max(0, int((game.get("end_ts", 0) - now_ms) / 1000))
                        if game.get("isActive") and remaining > 0:
                            try:
                                if label == game.get("current") and float(confidence) >= GAME_CONF_THRESH:
                                    game["score"] = int(game.get("score", 0)) + 1
                                    game["prev"] = game.get("current")
                                    pick_next_action(game)
                            except Exception:
                                # defensive: do not let game logic break inference flow
                                pass
                        if game.get("isActive") and remaining <= 0:
                            game["isActive"] = False

                        resp = {"label": label, "confidence": float(confidence), "timestamp": now_ms, "game": game_payload(game)}
                        await websocket.send_text(json.dumps(resp))
                except Exception as e:
                    await websocket.send_text(json.dumps({"error": "inference_failed", "detail": str(e)}))
                    # do not break; continue processing subsequent frames

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        pose.close()


def load_model_checkpoint(ckpt_path: str):
    """
    Load checkpoint using torch.load, build TCN and return (model, actions, in_ch).
    Exits with clear message on failure.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint '{ckpt_path}': {e}", file=sys.stderr)
        sys.exit(1)

    CLASSES = ckpt.get("classes")
    if CLASSES is None:
        print(f"ERROR: checkpoint '{ckpt_path}' missing 'classes' key", file=sys.stderr)
        sys.exit(1)

    in_ch = ckpt.get("in_channels", 66)
    model = TCN(in_ch, len(CLASSES))
    try:
        model.load_state_dict(ckpt["state_dict"])
    except Exception as e:
        # try to be helpful but continue (sometimes checkpoints store module prefixes)
        try:
            # strip "module." prefixes
            state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
            model.load_state_dict(state)
        except Exception:
            print(f"ERROR: Failed to load state_dict from '{ckpt_path}': {e}", file=sys.stderr)
            sys.exit(1)

    model.eval()
    return model, CLASSES, in_ch


# Load model at FastAPI startup so `uvicorn aiModel.webapp.backend:app` works.
# Uses environment variables AI_MODEL_PATH and AI_SEQ_LEN if provided.
@app.on_event("startup")
async def startup_event():
    """
    Attempt to load the checkpoint on application startup. If loading fails we
    set app.state.model to None (the websocket handler will return an explicit
    error "model_not_loaded"). This avoids requiring users to run the script
    directly and makes `uvicorn ...` work.
    """
    model_path = os.environ.get("AI_MODEL_PATH", "aiModel/models/ai_charades_tcn2.pt")
    try:
        seq_len = int(os.environ.get("AI_SEQ_LEN", "48"))
    except Exception:
        seq_len = 48

    # Default state
    app.state.model = None
    app.state.actions = None
    app.state.in_channels = None
    app.state.seq_len = seq_len

    logger.info(f"Startup: attempting to load model from '{model_path}' (seq_len={seq_len})")
    try:
        # load_model_checkpoint may call sys.exit on fatal errors; catch SystemExit
        model, classes, in_ch = load_model_checkpoint(model_path)
        app.state.model = model
        app.state.actions = classes
        app.state.in_channels = in_ch
        app.state.seq_len = seq_len
        logger.info(f"Model loaded on startup: '{model_path}' classes={len(classes)} in_ch={in_ch}")
    except SystemExit as e:
        logger.error(f"Model failed to load on startup (SystemExit): {e}. WebSockets will return model_not_loaded until fixed.")
    except Exception as e:
        logger.exception(f"Unexpected error loading model on startup: {e}. App will continue without a model.")


def parse_args():
    p = argparse.ArgumentParser(description="FastAPI backend for AI Charades webcam webapp")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind")
    p.add_argument("--port", type=int, default=8000, help="Port to bind")
    p.add_argument("--model", default="aiModel/models/ai_charades_tcn2.pt", help="Path to model checkpoint")
    p.add_argument("--seq_len", type=int, default=48, help="Sequence length for sliding window")
    return p.parse_args()


def main():
    args = parse_args()
    model, classes, in_ch = load_model_checkpoint(args.model)
    # attach to app state for use in websocket handlers
    app.state.model = model
    app.state.actions = classes
    app.state.in_channels = in_ch
    app.state.seq_len = args.seq_len

    logger.info(f"Loaded model='{args.model}' classes={len(classes)} in_ch={in_ch} seq_len={args.seq_len}")
    uvicorn.run("aiModel.webapp.backend:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()