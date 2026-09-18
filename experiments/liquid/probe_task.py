"""Real-physics container retention, controller construction and reset replay.

This is NOT an expert episode or a policy evaluation. No camera overrides.
"""

import argparse
import copy
import json
from pathlib import Path
import random
import traceback

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("name")
parser.add_argument("--output-root", default="outputs/pour_liquid/all_tasks_probes")
args = parser.parse_args()
app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from omegaconf import OmegaConf

from controllers.atomic_actions.atomic_base_controller import AtomicBaseController
from factories.controller_factory import create_controller
from factories.task_factory import create_task
from robots.franka.franka import Franka
from robots.franka.rmpflow_controller import RMPFlowController
from robots.ridgebase_franka.ridgebase import Ridgebase
from utils.object_utils import ObjectUtils

cfg = OmegaConf.load(f"config/{args.name}.yaml")
cfg.mode = "collect"
cfg.max_episodes = 1
cfg.liquid = OmegaConf.merge(OmegaConf.load("config/liquid/beaker.yaml"), cfg.get("liquid", {}))
out = Path(args.output_root) / args.name
out.mkdir(parents=True, exist_ok=True)
cfg.multi_run.run_dir = str(out / "run")
np.random.seed(901)
random.seed(901)
world = World(stage_units_in_meters=1, physics_prim_path="/physicsScene", backend="numpy")
stage = omni.usd.get_context().get_stage()
add_reference_to_stage(str(Path(cfg.usd_path).resolve()), "/World")
# This diagnostic deliberately loads only the two in-scope robot classes.
# It does not repair or hide missing unrelated registrations in the main entry.
robot_class = {"franka": Franka, "ridgebase": Ridgebase}[cfg.robot.type]
robot = robot_class(position=np.array(cfg.robot.position))
ObjectUtils.get_instance(stage)
report = {"name": args.name, "test": "physical_retention_and_initial_state_replay", "passed": False}
controller = None
try:
    task = create_task(cfg.task_type, cfg=cfg, world=world, stage=stage, robot=robot)
    RMPFlowController.ignore_robot_state_updates = bool(getattr(cfg, "collect_position_only", False))
    AtomicBaseController.record_commanded_gripper = bool(getattr(cfg, "collect_position_only", False))
    AtomicBaseController.randomization_enabled = False
    controller = create_controller(cfg.controller_type, cfg=cfg, robot=robot)
    task.reset()
    snapshot = copy.deepcopy(task._episode_init_state)
    initial = np.asarray(task.liquid.particles.GetPositionsAttr().Get()).copy()
    runs = []
    for mode in ("collect_init", "replay_init"):
        if mode == "replay_init":
            task.reset_with_init_state(snapshot)
            np.testing.assert_array_equal(np.asarray(task.liquid.particles.GetPositionsAttr().Get()), initial)
        for _ in range(1200):
            world.step(render=False)
        state = None
        for _ in range(task.WARMUP_FRAMES + 1):
            world.step(render=True)
            state = task.step()
            if state is not None:
                break
        assert state is not None, "Camera/state observation not ready after the retention run"
        metrics = task.liquid.observe()
        cohorts = metrics if task.liquid.routes else {"pour": metrics}
        if task.liquid.routes:
            initial_routes = {route.name for route in task.liquid.routes if route.source == route.initial_source}
        else:
            initial_routes = {"pour"}
        retained = {name: row["source_fraction"] for name, row in cohorts.items() if name in initial_routes}
        assert all(value >= 0.99 for value in retained.values()), f"Resting source leaked: {retained}"
        assert np.linalg.norm(np.asarray(task.liquid.particles.GetPositionsAttr().Get()) - initial) > 0.01
        controller.state = state
        if cfg.task_type == "pick_pour":
            # Normal controller.step() captures these before checking success.
            # This no-actions probe initializes them without executing a pick.
            controller.initial_position = state["object_position"]
            controller.initial_size = state["object_size"]
            controller.initial_quaternion = state["object_quaternion"]
        if getattr(controller, "liquid_monitor", None) is not None:
            controller.liquid_monitor.update(state)
        assert not controller._check_success(), "An unpoured initial scene passed success"
        if cfg.controller_type == "mobile_pour":
            controller._record_liquid_hold(state, 5)
            assert controller.data_collector.temp_actions[-1].shape == (11,)
            assert np.isfinite(controller.data_collector.temp_actions[-1]).all()
        runs.append({"mode": mode, "retained": retained})
    # Force a missing-snapshot replay to fail, never silently seed old data.
    legacy = copy.deepcopy(snapshot)
    legacy["extra"].pop("liquid")
    try:
        task.reset_with_init_state(legacy)
    except ValueError:
        pass
    else:
        raise AssertionError("Legacy no-liquid replay was not rejected")
    report.update(passed=True, particles=len(initial), runs=runs, legacy_rejected=True)
except Exception as error:
    report["error"] = f"{type(error).__name__}: {error}"
    report["traceback"] = traceback.format_exc()
    raise
finally:
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("LIQUID_PROBE", json.dumps(report), flush=True)
    if controller is not None:
        controller.close()
    app.close()
