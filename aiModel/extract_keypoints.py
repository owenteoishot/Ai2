import os, cv2, json
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Normalise keypoints to ignore height/position
def normalize_keypoints(pts):
    left_hip, right_hip = pts[23], pts[24]
    center = (left_hip + right_hip) / 2
    pts -= center
    left_shoulder, right_shoulder = pts[11], pts[12]
    scale = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6
    pts /= scale
    return pts[:, :2]  # only x, y

# Process a single video and save JSON
def process_video(video_path, output_path, label):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            pts = np.array([[l.x, l.y, l.z] for l in result.pose_landmarks.landmark])
            pts_norm = normalize_keypoints(pts)
            sequence.append(pts_norm.flatten().tolist())

    cap.release()
    pose.close()

    with open(output_path, "w") as f:
        json.dump({"label": label, "frames": sequence}, f, indent=2)
    print(f"✅ Saved {output_path} ({len(sequence)} frames)")

# Main loop — process all videos
INPUT_DIR = "dataset_videos"
OUTPUT_DIR = "pose_keypoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in os.listdir(INPUT_DIR):
    folder = os.path.join(INPUT_DIR, label)
    if not os.path.isdir(folder): continue

    for filename in os.listdir(folder):
        if not filename.endswith(".mp4"): continue

        video_path = os.path.join(folder, filename)
        out_path = os.path.join(OUTPUT_DIR, f"{label}_{filename.replace('.mp4','.json')}")
        process_video(video_path, out_path, label)
