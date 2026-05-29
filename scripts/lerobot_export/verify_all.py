"""Load every converted level1 dataset with the matching lerobot library
and report shape / length / feature integrity.

Run with EITHER:
  - main env (lerobot 0.4.4, v3.0)    → verify v30/<task>
  - /tmp/venv_lerobot21 (lerobot 0.3.3, v2.1)  → verify v21/<task>

CLI: python verify_all.py {v21|v30}
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent / "test_outputs" / "lerobot"
TASKS = [
    "level1_pick", "level1_place", "level1_pour", "level1_press",
    "level1_shake", "level1_stir", "level1_open_door", "level1_close_door",
    "level1_open_drawer", "level1_close_drawer", "level1_close_centrifuge",
]


def verify(tag: str):
    base = ROOT / tag
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print(f"\n=== Verifying {tag} ===")
    fails = 0
    for t in TASKS:
        dst = base / t
        try:
            ds = LeRobotDataset(repo_id=f"local/{tag}_{t}", root=str(dst),
                                video_backend="pyav") if tag == "v30" else \
                 LeRobotDataset(repo_id=f"local/{tag}_{t}", root=str(dst))
            sample = ds[0]
            keys = sorted(ds.features.keys())
            cams = [k for k in keys if k.startswith("observation.images.")]
            print(f"  ✅ {t:<28} eps={ds.num_episodes:>3}  frames={len(ds):>6}  "
                  f"state={tuple(sample['observation.state'].shape)}  "
                  f"action={tuple(sample['action'].shape)}  cams={len(cams)}  "
                  f"task='{sample['task'][:40]}...'")
        except Exception as e:
            print(f"  ❌ {t}: {type(e).__name__}: {str(e)[:120]}")
            fails += 1
    print(f"  total: {len(TASKS) - fails}/{len(TASKS)} passed")
    return fails


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v30"
    sys.exit(verify(tag))
