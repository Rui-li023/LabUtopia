# LabUtopia - Development Guide

A high-fidelity chemistry lab simulation benchmark for scientific embodied agents,
built on NVIDIA Isaac Sim. A Franka Panda robot performs tasks of increasing complexity
across 5 difficulty levels.

## Project Structure

```
main.py                         # Entry point: Hydra config → factory construction → sim loop
train.py / train-muilt.py       # Policy training (Diffusion UNet / ACT)
assets/                         # USD scene files
config/                         # Hydra YAML configs (level{1-5}_{TaskName}.yaml)
controllers/
  base_controller.py            # ABC base for all task controllers
  {action}_controller.py        # Per-task controllers (pick, pour, stir, ...)
  atomic_actions/               # Low-level state-machine controllers (pick, place, pour, ...)
  inference_engines/            # Local (PyTorch) / Remote (OpenPI) inference
  robot_controllers/            # Trajectory controller, gripper manager, RMPFlow wrapper
tasks/
  base_task.py                  # ABC base for all task environments
  single_object_task.py         # Single-object base (PickTask, PlaceTask, ...)
  dual_object_task.py           # Dual-object base (PickPlaceTask, PickPourTask, ...)
  {action}_task.py              # Per-task environments
factories/                      # Registry-based factories: task, controller, robot, collector
robots/                         # Franka, Ridgebase robot definitions
policy/                         # ML models: Diffusion UNet, ACT, vision encoders
data_collectors/                # HDF5 episode recording (DataCollector, ActionStateDataCollector)
utils/                          # ObjectUtils (singleton), camera, replay data loader
scripts/                        # Data conversion & dataset merge utilities
```

## Architecture & Patterns

### Three Operating Modes

Set via `cfg.mode` in YAML config:

- **collect** — scripted atomic actions → `DataCollector` → HDF5
- **infer** — trained policy model → `InferenceEngine` → `TrajectoryController`
- **replay** — replay recorded episodes from HDF5 deterministically

### Factory Pattern

All factories use the same registry pattern (`_registry` dict, no inheritance):

```python
_registry: Dict[str, Type] = {}
def register_X(name, cls): _registry[name] = cls
def create_X(name, *args, **kwargs): return _registry[name](*args, **kwargs)
```

Registry keys in config: `task_type` and `controller_type` must match (e.g. `"pick"`, `"pickplace"`).

### Task / Controller Separation

- **Task** (`tasks/`) — owns the scene: spawns objects, manages cameras, returns `state` dict
- **Controller** (`controllers/`) — owns the robot actions: receives `state`, returns `(action, done, is_success)`

They are created independently by their factories and connected in `main.py`.

### Controller Subclass Contract

Every controller extends `BaseController` and must implement:

```python
@abstractmethod
def _step_collect(self, state) -> Tuple[Any, bool, bool]: ...

@abstractmethod
def _step_infer(self, state) -> Tuple[Any, bool, bool]: ...

@abstractmethod
def _check_success(self) -> bool: ...
```

- `_step_replay` is fully implemented in `BaseController` — subclasses do NOT override it.
- `self.state` is set by `BaseController.step()` before dispatching — subclasses can use it freely.
- Success is tracked via `check_success_counter >= REQUIRED_SUCCESS_STEPS` (default 60).
- `_last_failure_reason` is always `str` (use `""` for no failure, never `None`).

### Task Subclass Contract

Every task extends `BaseTask` (or `SingleObjectTask` / `DualObjectTask`) and must implement:

```python
def step(self) -> Optional[Dict[str, Any]]: ...
```

Key base methods available:
- `setup_cameras()`, `setup_objects()`, `setup_materials()`
- `reset()` / `reset_with_init_state(init_state)`
- `get_basic_state_info(object_path, target_path=None)` — builds the standard state dict

## Naming Conventions

| Entity | Pattern | Example |
|--------|---------|---------|
| Task file | `{action}_task.py` | `pick_task.py`, `pickplace_task.py` |
| Controller file | `{action}_controller.py` | `pick_controller.py` |
| Task class | `{Action}Task` | `PickTask`, `PickPlaceTask` |
| Controller class | `{Action}TaskController` | `PickTaskController` |
| Config file | `level{N}_{TaskName}.yaml` | `level3_HeatLiquid.yaml` |
| Atomic action | `controllers/atomic_actions/{action}_controller.py` | Separate namespace |
| Camera obs key | `{camera_name}_{image_type}` | `camera_1_rgb` |
| Episode file | `episode_{NNNN}.h5` | `episode_0003.h5` |
| USD prim path | `/World/{object_name}` | `/World/conical_bottle02` |

## Config Schema (Hydra YAML)

```yaml
name: Level1_pick
task_type: "pick"               # must match factory registry key
controller_type: "pick"         # must match factory registry key
mode: "collect"                 # collect | infer | replay
usd_path: "assets/..."
max_episodes: 50

task:
  max_steps: 800
  obj_paths:                    # objects with position randomization ranges
    - path: "/World/object"
      position_range: {x: [...], y: [...], z: [...]}

cameras:                        # list of camera configs
  - prim_path: "/World/Camera1"
    name: "camera_1"
    resolution: [256, 256]
    image_type: "rgb"           # rgb | depth | pointcloud | rgb+depth | segmentation

robot:
  type: "franka"                # franka | ridgebase

collector:
  type: "default"               # default | mock | action_state
  compression: gzip

infer:
  type: "local"                 # local | remote
  obs_names: {...}
  is_test_material: false       # OOD generalization flag (level3+)

replay:
  dataset_path: ""
```

## Difficulty Levels

| Level | Scope | Example Tasks |
|-------|-------|---------------|
| L1 | Single atomic action | pick, place, press, shake, stir, open, close |
| L2 | Multi-step composed | HeatLiquid, PourLiquid, ShakeBeaker |
| L3 | Generalization (OOD materials/objects) | Same as L1/L2 with `test_materials` |
| L4 | Long-horizon sequences | CleanBeaker, DeviceOperation, LiquidMixing |
| L5 | Mobile manipulation | Navigation, MobilePickPlace |

## Coding Standards

- **Type hints**: All public methods must have return type annotations.
- **Abstract methods**: Use `@abstractmethod` decorator (not bare `raise NotImplementedError`).
- **Properties**: Use `@property` for attribute-like accessors (e.g. `episode_num`).
- **Imports**: Group as stdlib → third-party → project-internal, separated by blank lines.
  Move all imports to file top level (no inline `import` inside methods).
- **Sentinel values**: Use consistent types — `_last_failure_reason` is always `str` (`""`), never `None`.
- **OmegaConf resolvers**: Guard with `OmegaConf.has_resolver()` before registering.
- **Singleton**: `ObjectUtils` is accessed via `ObjectUtils.get_instance()`.
- **Logging**: Use `loguru.logger` (`logger.info`, `logger.warning`, `logger.success`).

## Key Dependencies

- NVIDIA Isaac Sim 5.1 (isaacsim runtime)
- PyTorch (GPU inference & training)
- Hydra / OmegaConf (config management)
- h5py (HDF5 dataset I/O)
- loguru (logging)
- openpi-client (remote inference, vendored in `packages/`)
