import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import os
import argparse
import sys

# Minimal TCN model (unchanged)
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

    def forward(self, x):  # x: (B,F,T)
        z = self.net(x).squeeze(-1)
        return self.fc(z)

def normalize_keypoints(pts):
    """
    Normalize keypoints: center at hips midpoint and scale by shoulder distance.
    Returns first two coords. Kept small and identical to original behavior.
    """
    left_hip, right_hip = pts[23], pts[24]
    center = (left_hip + right_hip) / 2
    pts = pts.copy()
    pts -= center
    left_shoulder, right_shoulder = pts[11], pts[12]
    scale = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6
    pts /= scale
    return pts[:, :2]

def main():
    parser = argparse.ArgumentParser(description="Run webcam demo for AI Charades")
    parser.add_argument("--camera", type=int, default=0, help="Camera device id")
    parser.add_argument("--model", type=str, default="models/ai_charades_tcn2.pt", help="Path to model checkpoint")
    parser.add_argument("--seq_len", type=int, default=48, help="Sequence length")
    args = parser.parse_args()

    # ---------- Load model (wrapped with error handling) ----------
    ckpt_path = args.model
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        # Clear error message and exit with non-zero code so CI/dev can detect failure
        print(f"ERROR: Failed to load checkpoint '{ckpt_path}': {e}", file=sys.stderr)
        sys.exit(1)

    CLASSES = ckpt["classes"]
    in_ch = ckpt.get("in_channels", 66)  # fallback if missing

    model = TCN(in_ch, len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    actions = CLASSES

    # Setup MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Webcam loop
    cap = cv2.VideoCapture(args.camera)  # use CLI camera id
    sequence = []
    SEQUENCE_LENGTH = args.seq_len

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        # validate mediapipe detection: skip frame when no landmarks
        if not result.pose_landmarks:
            # do not crash; show frame and continue
            cv2.imshow("AI Charades", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        lm = result.pose_landmarks.landmark
        # validate landmark count (expect 33) before indexing; skip frame on mismatch
        if len(lm) != 33:
            cv2.imshow("AI Charades", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        pts = np.array([[l.x, l.y, l.z] for l in lm])
        pts_norm = normalize_keypoints(pts)
        sequence.append(pts_norm.flatten())

        if len(sequence) > SEQUENCE_LENGTH:
            sequence = sequence[-SEQUENCE_LENGTH:]

        if len(sequence) == SEQUENCE_LENGTH:
            x = np.array(sequence, dtype=np.float32)
            x = torch.tensor(x).unsqueeze(0)  # (1, seq_len, features)
            x = x.permute(0, 2, 1)            # (1, features, seq_len)

            with torch.no_grad():
                out = model(x)
                pred = torch.softmax(out, dim=1)
                action_id = torch.argmax(pred).item()
                confidence = pred[0, action_id].item()

            action = actions[action_id]
            cv2.putText(frame, f"{action} ({confidence:.2f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("AI Charades", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
