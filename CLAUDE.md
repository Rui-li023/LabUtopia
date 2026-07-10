# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LabUtopia (NeurIPS 2025) is a high-fidelity chemistry lab simulation benchmark for scientific
embodied agents, built on NVIDIA Isaac Sim 5.1. A Franka Panda robot (Ridgebase mobile base at
Level 5) performs tasks of increasing complexity across 5 difficulty levels.

## Environment & Commands

Conda env `labutopia` (Python 3.11). Install order matters:
PyTorch 2.9.0 (cu126) → `isaacsim[all,extscache]==5.1.0` → `pip install -r requirements.txt`.
Isaac Sim does not run on A100/A800 — RTX-class GPU required.

```bash
# Run a task (collect / infer / replay is chosen by `mode:` inside the YAML)
python main.py --config-name=level1_pick            # config/<name>.yaml
python main.py --config-name=level1_pick --no-video # skip mp4 encoding
# Other flags: --headless, --backend numpy|gpu, --config-dir <dir>

# Lint (required before PRs; line-length 120, excludes packages/ outputs/ assets/)
ruff check
ruff format --check

# Tests — standalone scripts, NOT pytest; they launch Isaac Sim so run them one at a time
python tests/test_config_files.py       # sweeps all collect configs at max_episodes=8
python tests/test_single_config.py      # single config runner

# Export a collected run to LeRobot format
python -m scripts.lerobot_export.cli --src <run_dir> --dst <out_dir> --version v2.1  # or v3.0
```

Outputs land in `outputs/${mode}/${date}/${time}_${name}/`; collected episodes are
`episode_{NNNN}.h5` plus per-camera mp4 videos.

- **Never run multiple `main.py` (Isaac Sim) processes in parallel.** A single Isaac Sim
  instance already saturates the GPU; concurrent runs cause physics instability, OOM, and
  watchdog kills. Run regression tests sequentially — even when they look independent.
- `policy/` contains four vendored VLA repos as git submodules (Isaac-GR00T, lerobot,
  lingbot-vla, openpi) — training happens inside those submodules. The in-tree Diffusion
  UNet / ACT training code that `train.py` / `train-muilt.py` expect (`policy/config/`)
  was removed in that refactor, so those entry points are stale.

## Project Structure

```
main.py                         # Entry point: Hydra config → factory construction → sim loop
assets/                         # USD scene files
config/                         # Hydra YAML configs (level{1-5}_{task_name}.yaml)
controllers/
  base_controller.py            # Base for all task controllers
  {action}_controller.py        # Per-task controllers (pick, pour, stir, ...)
  atomic_actions/               # Low-level state-machine controllers (pick, place, pour, ...)
  inference_engines/            # local (PyTorch checkpoint) / remote (WebSocket, OpenPI-style)
  robot_controllers/            # Trajectory controller, gripper manager, RMPFlow wrapper
tasks/
  base_task.py                  # ABC base for all task environments
  single_object_task.py         # Single-object base (PickTask, PlaceTask, ...)
  dual_object_task.py           # Dual-object base (PickPlaceTask, PickPourTask, ...)
factories/                      # Registry-based factories: task, controller, robot, collector
robots/                         # Franka, Ridgebase, Piper robot definitions
policy/                         # Git submodules: Isaac-GR00T, lerobot, lingbot-vla, openpi
data_collectors/                # HDF5 episode recording (default | action_state | mobile | mock)
utils/                          # ObjectUtils (singleton), camera, replay data loader
scripts/                        # Campaign runners, LeRobot export, eval harnesses (see below)
docs/                           # Local knowledge base: campaign/eval/debug reports (not shipped)
tests/                          # Standalone validation scripts (launch Isaac Sim subprocesses)
packages/                       # Vendored openpi-client (pip install -e packages/openpi-client)
```

## Architecture & Patterns

### Three Operating Modes

Set via `cfg.mode` in YAML config:

- **collect** — scripted atomic actions → `DataCollector` → HDF5
- **infer** — trained policy model → `InferenceEngine` → `TrajectoryController`
- **replay** — replay recorded episodes from HDF5 deterministically

### Factory Pattern

All four factories (`factories/{task,controller,robot,collector}_factory.py`) use the same
registry pattern (`_registry` dict, no inheritance):

```python
_registry: Dict[str, Type] = {}
def register_X(name, cls): _registry[name] = cls
def create_X(name, *args, **kwargs): return _registry[name](*args, **kwargs)
```

`task_type` and `controller_type` in config must match registry keys (e.g. `"pick"`,
`"pick_place"`, `"mobile_pick"`). Collector keys: `default | mock | action_state | mobile`.
Robot keys: `franka | ridgebase | piper`.

### Task / Controller Separation

- **Task** (`tasks/`) — owns the scene: spawns objects, manages cameras, returns `state` dict
- **Controller** (`controllers/`) — owns the robot actions: receives `state`, returns `(action, done, is_success)`

They are created independently by their factories and connected in `main.py`.

### Controller Subclass Contract

Every controller extends `BaseController` and must implement:

```python
def _step_collect(self, state) -> Tuple[Any, bool, bool]: ...
def _step_infer(self, state) -> Tuple[Any, bool, bool]: ...
def _check_success(self) -> bool: ...
```

(In `BaseController` these currently raise `NotImplementedError` rather than using
`@abstractmethod` — new abstract methods should use the decorator per coding standards.)

- `_step_replay` is fully implemented in `BaseController` — subclasses do NOT override it.
  **Exception:** Level-5 mobile controllers (`MobileManipControllerBase` subclasses) override it,
  because recorded actions are 11-dim (base x/y/θ + 7 arm + gripper) and are applied directly
  as per-frame position commands on all 12 Ridgebase DOFs instead of via the Franka
  trajectory controller.
- `self.state` is set by `BaseController.step()` before dispatching — subclasses can use it freely.
- Success is tracked via `check_success_counter >= REQUIRED_SUCCESS_STEPS` (default 60).
- `_last_failure_reason` is always `str` (use `""` for no failure, never `None`).

### Task Subclass Contract

Every task extends `BaseTask` (or `SingleObjectTask` / `DualObjectTask`) and must implement:

```python
@abstractmethod
def step(self) -> Optional[Dict[str, Any]]: ...
```

Key base methods available:
- `setup_cameras()`, `setup_objects()`, `setup_materials()`
- `reset()` / `reset_with_init_state(init_state)`
- `get_basic_state_info(object_path, target_path=None)` — builds the standard state dict

### Deterministic Replay & `collect_position_only`

Each episode records an `init_state` dict (object poses/materials, camera poses, robot init
joints and world position) in every step's state; `reset_with_init_state(init_state)` restores
the exact scene for replay.

`collect_position_only: true` (top-level config flag, read in `main.py`) makes collection use
the same pure position-based control law as replay/inference: RMPFlow tracks its internal
virtual robot (`ignore_robot_state_updates = True`), velocity/effort components are stripped
from applied actions, and the gripper channel records the *commanded* position instead of the
measured one. Without it, phase transitions in scripted multi-step controllers drift on replay.
Most L1–L5 configs enable it.

## Naming Conventions

| Entity | Pattern | Example |
|--------|---------|---------|
| Task file | `{action}_task.py` | `pick_task.py`, `pick_place_task.py` |
| Controller file | `{action}_controller.py` | `pick_controller.py` |
| Task class | `{Action}Task` | `PickTask`, `PickPlaceTask` |
| Controller class | `{Action}TaskController` | `PickTaskController` |
| Config file | `level{N}_{task_name}.yaml` | `level3_heat_liquid.yaml` |
| Atomic action | `controllers/atomic_actions/{action}_controller.py` | Separate namespace |
| Camera obs key | `{camera_name}_{image_type}` | `camera_1_rgb` |
| Episode file | `episode_{NNNN}.h5` | `episode_0003.h5` |
| USD prim path | `/World/{object_name}` | `/World/conical_bottle02` |

**Convention:** filenames, config names, and registry keys (`task_type` / `controller_type`)
are `snake_case`; only class names use `PascalCase`. Multi-word names are split with
underscores (e.g. `open_close`, `clean_beaker`, `open_transport_pour`).

## Config Schema (Hydra YAML)

```yaml
name: level1_pick
task_type: "pick"               # must match factory registry key
controller_type: "pick"         # must match factory registry key
mode: "collect"                 # collect | infer | replay
usd_path: "assets/..."
max_episodes: 50                # collect: counts SUCCESSFUL episodes
collect_position_only: true     # position-only control law (see above)

task:
  max_steps: 800
  obj_paths:                    # objects with position randomization ranges
    - path: "/World/object"
      position_range: {x: [...], y: [...], z: [...]}
  material_paths: [...]         # optional (L3 OOD materials)

cameras:                        # list of camera configs
  - prim_path: "/World/Camera1"
    name: "camera_1"
    resolution: [256, 256]
    image_type: "rgb"           # rgb | depth | pointcloud | rgb+depth | segmentation

robot:
  type: "franka"                # franka | ridgebase | piper
  gripper:                      # optional
    control_mode: "position"    # position | force

collector:
  type: "default"               # default | mock | action_state | mobile
  compression: gzip

infer:
  type: "local"                 # local | remote (WebSocket, OpenPI-style server)
  obs_names: {...}
  is_test_material: false       # OOD generalization flag (level3+)
  # local: policy_model_path / policy_config_path / normalizer_path
  # remote: host / port / n_obs_steps / action_chunk_len / timeout / max_retries

replay:
  dataset_path: ""              # dir containing episode_*.h5
  episode_indices: []           # optional subset
```

## Difficulty Levels

| Level | Scope | Tasks |
|-------|-------|-------|
| L1 | Single atomic action | pick, place, pour, press, shake, stir, open/close door & drawer, close_centrifuge |
| L2 | Multi-step composed | heat_liquid, pour_liquid, shake_beaker, stir_glassrod, transport_beaker, open_close, flask_to_cork, flask_to_triangle, stopper_to_flask, pipette_to_rack |
| L3 | Generalization (OOD materials/objects) | pick, press, open, pour_liquid, heat_liquid, transport_beaker with `test_materials` |
| L4 | Long-horizon sequences | clean_beaker, device_operation, liquid_mixing, open_transport_pour |
| L5 | Mobile manipulation (Ridgebase) | navigation, close_pick, close_pick_place, far_pick, far_transport_place |

### Level-5 status & design knobs (2026-07-09)

All four L5 tasks collect at 100 episodes (attempt rate close_pick 87% / far_pick 95% /
close_pick_place 78% / far_transport_place 88%); close_pick replays 94/100 in real physics.
Task-specific config keys (under `task:`), unique to the mobile tasks:

- `spawn.mode`: `near` (close tasks) spawn ~1 m on the object→dock axis facing the object;
  `far` samples free-space with A\* path ≥ `spawn.min_path_length`.
- `dock_standoff`, `DOCK_X_OFFSET` (−0.08, in `MobilePickTask`), `FINAL_NAV_ANGLE` (π/2−0.044):
  park the base in the validated grasp band (object 3–10 cm on the arm's right, heading 86–88°).
- `dock_jitter: [x, y]` — per-episode ±jitter (m) on the stop pose for VLA robustness (default off;
  L5 uses ±0.03). `grasp_ee_euler_deg` `[-115,90,0]` (side grasp), `place_ee_euler_deg` `[-110,90,0]`.
- **Facing** is set via the base revolute joint, NOT the root orientation (rotating the root rotates
  the prismatic-joint frame the controller commands in world coords → sideways "crab" drift).
- **Short carry** (`< CARRY_HOLD_MAX_DIST` 2 m, close place) crabs sideways holding the bench
  heading (`RidgebaseController.set_waypoints(hold_heading=True)`); long carries face travel dir.
- Nav is pure-pursuit with a low-passed heading (`RidgebaseController`); `max_angular_speed` 0.12,
  `k_p_angular` 1.5 — earlier values saturated the yaw and shook the cameras.

**LeRobot export** (`scripts/lerobot_export/cli.py --base-action body_delta`): base action dims 0:3
become per-step BODY-frame deltas `[forward, lateral, dtheta]` (arm dims stay absolute joints).
`info.json.base_action` records it; the inference executor must integrate deltas onto the current
base pose (replay path: `replay.base_delta_actions: true`, closed-loop). Delivered per-task (no
merge) to cluster `v21/level5/<task>/`.

**Full-body VLA inference** (2026-07-10): `infer.type: remote_mobile` →
`MobileRemoteInferenceEngine` (11-dim state, no Franka trajectory controller); the mobile
controllers' `_step_infer` applies the predicted 11-dim chunk via `_apply_action11`
(`infer.base_delta_actions: true` integrates base deltas closed-loop — same law as replay).
Gotcha: the mobile collector records state gripper (dim 10) as finger1×2 — the engine mirrors
this (`pose[10] *= 2`); sending raw finger1 halves the gripper state and kills grasping.
close_pick eval (20 ep, two metrics — nav progress = spawn→dock fraction closed, grasp success):
lingbot 0.71/15%, smolvla 0.84/10%, openpi 0.59/10%, gr00t 0.23/5%. Nav is learned; the grasp
(descend+close) is the universal bottleneck at 20k steps / 100 demos. Harness:
`scripts/level5_eval/` (serve.sh / tunnel.sh / metrics.sh); report `docs/level5_inference_report.md`.

## Scripts & Local Knowledge Base

- `scripts/lerobot_export/` — LeRobot v2.1/v3.0 export (`cli.py`, `merge_v21.py`,
  `verify_all.py`); one run dir per task, chunked episode layout.
- `scripts/level345_collect/`, `scripts/level23_campaign/` — batch collect/replay campaign
  runners; generate a temp Hydra config (`gen_config.py`) then run
  `python main.py --config-name _autorun`.
- `scripts/level23_eval/`, `scripts/level1_eval/` — remote VLA inference eval harnesses
  (OpenPI, LingBot, SmolVLA, GR00T) incl. cluster serve/submit and tunnel scripts.
- `docs/` — local (non-shipped) reports on collect/replay/inference campaigns; check here
  first for prior findings before re-debugging a task. Start with `docs/PROJECT_OVERVIEW.md`
  (项目通识): consolidated project overview, level status, durable lessons, and an index of
  which reports are current vs. superseded.

## Coding Standards

- **Type hints**: All public methods must have return type annotations.
- **Abstract methods**: Use `@abstractmethod` decorator (not bare `raise NotImplementedError`).
- **Properties**: Use `@property` for attribute-like accessors (e.g. `episode_num`).
- **Imports**: Group as stdlib → third-party → project-internal, separated by blank lines
  (ruff isort enforced). Move all imports to file top level (no inline `import` inside methods).
- **Sentinel values**: Use consistent types — `_last_failure_reason` is always `str` (`""`), never `None`.
- **OmegaConf resolvers**: Guard with `OmegaConf.has_resolver()` before registering.
- **Singleton**: `ObjectUtils` is accessed via `ObjectUtils.get_instance()`.
- **Logging**: Use `loguru.logger` (`logger.info`, `logger.warning`, `logger.success`).
