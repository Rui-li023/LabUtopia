"""Send recorded training frames to the remote policy and compare predicted
actions against the ground-truth actions saved during data collection.

Run inside the project root:
    python scripts/probe_remote_with_training.py \
        --episode outputs/collect/level1_pick/run/dataset/episode_0000 \
        --host 127.0.0.1 --port 18081 \
        --frames 0 100 300 500
"""

import argparse
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def read_frame(mp4_path: str, frame_idx: int, color: str = "rgb") -> np.ndarray:
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx} from {mp4_path}")
    if color == "rgb":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # else leave as BGR (test hypothesis that model trained on BGR)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 100, 300])
    ap.add_argument("--prompt", default="Pick up the conical bottle.")
    ap.add_argument("--color", choices=["rgb", "bgr"], default="rgb")
    args = ap.parse_args()

    h5_path = os.path.join(args.episode, os.path.basename(args.episode) + ".h5")
    mp4 = {
        "observation/image":          os.path.join(args.episode, "image.mp4"),
        "observation/wrist_image":    os.path.join(args.episode, "image_2.mp4"),
        "observation/wrist_image_2":  os.path.join(args.episode, "wrist_image.mp4"),
    }

    with h5py.File(h5_path, "r") as f:
        agent_pose = f["agent_pose"][:]   # (T, 8) — gripper width
        actions    = f["actions"][:]      # (T, 8) — gripper {0,1}
    T = len(agent_pose)
    print(f"episode length T={T}, prompt={args.prompt!r}")

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    print("server metadata:", client.get_server_metadata())

    for t in args.frames:
        if t >= T:
            print(f"[t={t}] out of range, skip")
            continue

        obs = {key: read_frame(path, t, args.color) for key, path in mp4.items()}
        for key, img in obs.items():
            print(f"[t={t}] {key}: shape={img.shape} dtype={img.dtype} "
                  f"min={img.min()} max={img.max()}")

        obs["observation/state"] = agent_pose[t].astype(np.float32)
        obs["prompt"] = args.prompt
        print(f"[t={t}] state={agent_pose[t]}")

        result = client.infer(obs)
        act = np.asarray(result.get("actions", result.get("action")))
        gt  = actions[t:t + len(act)]

        print(f"[t={t}] predicted action shape={act.shape}")
        print(f"        predicted[0]  = {act[0]}")
        print(f"        ground-truth  = {gt[0] if len(gt) else 'n/a'}")
        if len(gt) >= 1:
            n = min(len(act), len(gt))
            diff = np.abs(act[:n] - gt[:n]).mean(axis=0)
            print(f"        mean |Δ| over first {n} steps: {diff}")
        print(f"        predicted gripper[:,7] min={act[:,7].min():.3f} max={act[:,7].max():.3f}")
        print()


if __name__ == "__main__":
    main()
