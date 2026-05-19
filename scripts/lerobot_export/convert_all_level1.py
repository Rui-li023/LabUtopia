"""Convert all level1 collect100 datasets to BOTH lerobot v2.1 and v3.0.

Outputs to test_outputs/lerobot/{v21,v30}/<task>/.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = REPO / "test_outputs" / "level1_collect100"
DST_ROOT = REPO / "test_outputs" / "lerobot"
LOG_PATH = DST_ROOT / "convert.log"
DST_ROOT.mkdir(parents=True, exist_ok=True)

TASKS = [
    "level1_pick", "level1_place", "level1_pour", "level1_press",
    "level1_shake", "level1_stir", "level1_open_door", "level1_close_door",
    "level1_open_drawer", "level1_close_drawer", "level1_CloseCentrifuge",
]
VERSIONS = ["v2.1", "v3.0"]


def log(line: str):
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    only = sys.argv[1:] or TASKS
    for task in only:
        src = SRC_ROOT / task / "run"
        if not (src / "dataset").exists():
            log(f"❓ {task}: no dataset at {src}")
            continue
        for ver in VERSIONS:
            tag = ver.replace(".", "")  # v21 / v30
            dst = DST_ROOT / tag / task
            log(f"▶ {task} → {ver}  → {dst}")
            t0 = time.time()
            cp = subprocess.run(
                [sys.executable, "-m", "scripts.lerobot_export.cli",
                 "--src", str(src), "--dst", str(dst), "--version", ver],
                cwd=str(REPO), capture_output=True, text=True,
            )
            dur = time.time() - t0
            ok = cp.returncode == 0
            log(f"  {'✅' if ok else '❌'} {task}/{ver} in {dur/60:.1f} min (rc={cp.returncode})")
            if not ok:
                log(cp.stderr[-2000:])


if __name__ == "__main__":
    main()
