"""Derive a replay-mode config from a collect config, for figure rendering.

Some long-horizon tasks no longer collect reliably in the current tree, but their
validated 100-episode datasets are on disk.  Replaying one of those episodes puts
the arm through the identical motion, so a figure strip can still be rendered
from a real successful rollout.

Writes ``<out_dir>/<name>.yaml``; point ``experiments/main.py --config-dir`` at ``<out_dir>``.
"""

import argparse
import os

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="source config name, e.g. level4_device_operation")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument(
        "--config-path",
        help="explicit source YAML; prefer the config.yaml saved NEXT TO the "
        "dataset, so the replay uses the exact settings the episodes were "
        "recorded with rather than a since-edited config/ file",
    )
    parser.add_argument("--dataset", required=True, help="dir holding episode_XXXX/ sub-dirs")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", default=None, help="output config name (default: <config>_paper_replay)")
    parser.add_argument(
        "--episodes", type=int, nargs="*", default=None, help="episode indices to replay (default: the first one)"
    )
    args = parser.parse_args()

    if not args.config_path and not args.config:
        raise SystemExit("pass --config or --config-path")
    src = args.config_path or os.path.join(args.config_dir, f"{args.config}.yaml")
    cfg = OmegaConf.load(src)
    OmegaConf.set_struct(cfg, False)
    cfg.mode = "replay"
    cfg.replay = OmegaConf.create(
        {
            "dataset_path": os.path.abspath(args.dataset),
            "episode_indices": list(args.episodes) if args.episodes else [0],
            "max_episodes": 1,
        }
    )
    stem = args.config or os.path.splitext(os.path.basename(src))[0]
    name = args.name or f"{stem}_paper_replay"
    cfg.name = name
    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, f"{name}.yaml")
    # resolve=False keeps ${now:...} interpolations in hydra/multi_run intact.
    OmegaConf.save(cfg, dst, resolve=False)
    print(dst)


if __name__ == "__main__":
    main()
