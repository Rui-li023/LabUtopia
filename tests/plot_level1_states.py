"""Plot collected init-poses and joint trajectories for every level1 task.

Outputs two PNGs into test_outputs/level1_replay/:
  - init_poses_grid.png:  per-task 2D xy scatter of episode init positions
                          overlaid on the configured position_range.
  - agent_pose_grid.png:  per-task overlay of the 7 arm joint trajectories
                          across 5 episodes.
"""
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "test_outputs" / "level1_replay"

LEVEL1 = [
    "level1_pick", "level1_place", "level1_pour", "level1_press",
    "level1_shake", "level1_stir", "level1_open_door", "level1_close_door",
    "level1_open_drawer", "level1_close_drawer", "level1_CloseCentrifuge",
]


def latest_attempt_dir(task: str) -> Path | None:
    base = OUT_DIR / task
    if not base.exists():
        return None
    attempts = sorted([d for d in base.iterdir() if d.name.startswith("attempt_")])
    return attempts[-1] if attempts else None


def load_config_ranges(task: str) -> dict[str, dict]:
    cfg_path = REPO / "config" / f"{task}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    out = {}
    for entry in cfg.get("task", {}).get("obj_paths", []) or []:
        if isinstance(entry, str):
            continue
        pr = entry.get("position_range") or {}
        out[entry["path"]] = pr
    return out


def plot_init_poses():
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.flatten()
    for ax, task in zip(axes, LEVEL1):
        attempt = latest_attempt_dir(task)
        if attempt is None or not (attempt / "init_poses.json").exists():
            ax.set_title(f"{task}\n(no data)", fontsize=9)
            ax.axis("off")
            continue
        episodes = json.loads((attempt / "init_poses.json").read_text())
        ranges = load_config_ranges(task)

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(ranges), 1)))
        path_to_color = {p: colors[i] for i, p in enumerate(ranges.keys())}

        # Draw range boxes
        for path, pr in ranges.items():
            if not pr or "x" not in pr or "y" not in pr:
                continue
            x0, x1 = pr["x"]; y0, y1 = pr["y"]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, ec=path_to_color[path],
                                       lw=1.2, ls="--",
                                       label=path.split("/")[-1]))

        # Plot episode init positions
        for ep in episodes:
            for path, pos in zip(ep["paths"], ep["positions"]):
                c = path_to_color.get(path, "k")
                ax.scatter(pos[0], pos[1], color=c, s=30, edgecolor="black",
                           linewidth=0.4)

        ax.set_title(task, fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        if ranges:
            ax.legend(fontsize=7, loc="best")

    # hide extra axes
    for ax in axes[len(LEVEL1):]:
        ax.axis("off")

    fig.suptitle("Level1 episode initial object positions (xy) vs configured ranges",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "init_poses_grid.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def plot_agent_poses():
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=False)
    axes = axes.flatten()
    for ax, task in zip(axes, LEVEL1):
        attempt = latest_attempt_dir(task)
        if attempt is None:
            ax.set_title(f"{task}\n(no data)", fontsize=9)
            ax.axis("off")
            continue
        ds = attempt / "collect_run" / "dataset"
        h5s = sorted(ds.glob("episode_*/episode_*.h5"))
        if not h5s:
            ax.set_title(f"{task}\n(no h5)", fontsize=9)
            ax.axis("off")
            continue
        cmap = plt.cm.tab10(np.linspace(0, 1, 7))
        for h5p in h5s[:5]:
            with h5py.File(h5p, "r") as f:
                ap = f["agent_pose"][()]  # [T, 8]
                t = np.arange(ap.shape[0])
                for j in range(7):
                    ax.plot(t, ap[:, j], color=cmap[j], alpha=0.45, lw=0.7)
        ax.set_title(f"{task}  ({len(h5s)} eps)", fontsize=10)
        ax.set_xlabel("frame"); ax.set_ylabel("joint (rad)")
        ax.grid(True, alpha=0.3)

    # hide extras
    for ax in axes[len(LEVEL1):]:
        ax.axis("off")

    # one shared legend for joints
    handles = [plt.Line2D([0], [0], color=plt.cm.tab10(np.linspace(0, 1, 7))[j],
                          lw=2, label=f"joint{j+1}") for j in range(7)]
    fig.legend(handles=handles, ncol=7, loc="lower center", fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Level1 agent_pose (7 arm joints) — 5 episodes overlay per task",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = OUT_DIR / "agent_pose_grid.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def plot_gripper_states():
    """Per-task overlay of the gripper channel (agent_pose[:,7]) for each
    episode. Value convention from the collector: 0 = open, 1 = closed.
    Step transitions reveal pick (0→1) and release (1→0) events."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey=True)
    axes = axes.flatten()
    for ax, task in zip(axes, LEVEL1):
        attempt = latest_attempt_dir(task)
        if attempt is None:
            ax.set_title(f"{task}\n(no data)", fontsize=9)
            ax.axis("off")
            continue
        ds = attempt / "collect_run" / "dataset"
        h5s = sorted(ds.glob("episode_*/episode_*.h5"))
        if not h5s:
            ax.set_title(f"{task}\n(no h5)", fontsize=9)
            ax.axis("off")
            continue
        ep_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(h5s)))
        for ep_i, h5p in enumerate(h5s):
            with h5py.File(h5p, "r") as f:
                ap = f["agent_pose"][()]  # [T, 8]
            grip = ap[:, 7]
            ax.step(np.arange(grip.shape[0]), grip,
                    color=ep_colors[ep_i], lw=1.1, alpha=0.85,
                    where="post", label=f"ep{ep_i}")
        ax.set_title(f"{task}  ({len(h5s)} eps)", fontsize=10)
        ax.set_xlabel("frame")
        ax.set_ylabel("gripper (0=open, 1=closed)")
        ax.set_ylim(-0.15, 1.15)
        ax.set_yticks([0.0, 1.0])
        ax.grid(True, alpha=0.3)

    for ax in axes[len(LEVEL1):]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], color=plt.cm.viridis(0.15 + 0.7 * i / 4),
                          lw=2, label=f"ep{i}") for i in range(5)]
    fig.legend(handles=handles, ncol=5, loc="lower center", fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Level1 gripper state (agent_pose[:,7]) — 5 episodes overlay per task",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = OUT_DIR / "gripper_state_grid.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_init_poses()
    plot_agent_poses()
    plot_gripper_states()
