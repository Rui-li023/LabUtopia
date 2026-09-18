"""Generate a bounded, reproducible liquid test config outside production defaults."""

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("--base", default="level2_pour_liquid")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=901)
parser.add_argument("--replay")
parser.add_argument("--max-steps", type=int)
parser.add_argument("--close-up", action="store_true")
args = parser.parse_args()
cfg = OmegaConf.load(f"config/{args.base}.yaml")
cfg.mode = "replay" if args.replay else "collect"
cfg.seed = args.seed
cfg.max_episodes = args.episodes
cfg.action_randomization = False
if args.max_steps is not None:
    cfg.task.max_steps = args.max_steps
cfg.collector.record_precise_replay_actions = True
cfg.liquid = OmegaConf.merge({
    "enabled": True, "particle_spacing": 0.004, "fill_height": 0.025,
    "isosurface": True, "min_target_fraction": 0.8, "max_source_fraction": 0.15,
    "max_outside_fraction": 0.05, "settle_steps": 60,
}, cfg.get("liquid", {}))
if args.replay:
    cfg.replay.dataset_path = args.replay
    cfg.replay.max_episodes = args.episodes
if args.close_up:
    for camera in cfg.cameras:
        camera.resolution = [512, 512]
    camera = cfg.cameras[0]
    eye = np.array([0.85, -0.78, 1.25])
    focus = np.array([0.28, -0.12, 0.86])
    backward = eye - focus
    backward /= np.linalg.norm(backward)
    right = np.cross([0, 0, 1], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    camera.translation = eye.tolist()
    camera.orientation = Rotation.from_matrix(np.column_stack([right, up, backward])).as_quat()[[3, 0, 1, 2]].tolist()
    camera.focal_length = 22
output = Path("outputs/pour_liquid") / args.name
output.mkdir(parents=True, exist_ok=True)
cfg.multi_run.run_dir = str(output / "run")
cfg.hydra.run.dir = cfg.multi_run.run_dir
OmegaConf.save(cfg, output / "config.yaml")
print(output / "config.yaml")
