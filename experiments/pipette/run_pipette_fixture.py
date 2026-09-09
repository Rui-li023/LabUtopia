"""One-attempt fixed-initial-scene development probe using the normal policy.

By default only initial scene restoration differs from normal collect; explicit
development flags additionally change planning. No object pose is written
during actions. Original RNG continuation is unavailable; reproduce a
baseline failure with the same fixture/seed before attributing an improvement.
This is not independent acceptance. Output directories must be new.
"""

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
from functools import partial
from itertools import pairwise, product
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def with_contact_guard(original, contact_source, finished_phase):
    """Latch an unexpected contact across ALL phases, including handoffs.

    A failure is terminal before another command can be issued or a successful
    episode committed. Only the normal controller's failed-episode bookkeeping
    is reproduced; no object or robot state is written.
    """

    def step(active, state):
        contact = contact_source()
        if contact is None:
            return original(active, state)
        active.state = state
        active._last_failure_reason = f"Unexpected physical robot contact before next action: {contact}"
        active._last_success = False
        active.current_phase = finished_phase
        active.data_collector.clear_cache()
        logger.warning(active._last_failure_reason)
        return None, True, False

    return step


def read_arm_drive_properties(robot):
    """Snapshot solver drive settings without changing gains, limits or state."""
    properties = robot.dof_properties
    return {
        "type": "arm_drive_properties",
        "joint_names": list(robot.dof_names),
        "dof_properties": {name: properties[name].copy() for name in properties.dtype.names},
        "joint_friction": np.asarray(robot._articulation_view.get_friction_coefficients()).copy(),
        "enabled_self_collisions": bool(robot.get_enabled_self_collisions()),
        "measured_effort_semantics": "solver joint force projected onto DOF, not isolated motor torque",
    }


def read_arm_dynamics(robot, applied):
    """Copy observations BEFORE the controller issues the next action."""

    def snapshot(value):
        return None if value is None else np.asarray(value).copy()

    return {
        "measured_velocities": snapshot(robot.get_joint_velocities()),
        "measured_joint_efforts": snapshot(robot.get_measured_joint_efforts()),
        "applied_positions": snapshot(None if applied is None else applied.joint_positions),
        "applied_velocities": snapshot(None if applied is None else applied.joint_velocities),
        "applied_efforts": snapshot(None if applied is None else applied.joint_efforts),
    }


def smooth_joint_path(path, steps_per_radian=400):
    """Slow rest-to-rest joint interpolation for a bounded simulation probe."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError("Expected at least two finite seven-joint waypoints")
    if not np.isfinite(steps_per_radian) or not 200 <= steps_per_radian <= 800:
        raise ValueError("Interpolation rate must be finite and in [200, 800] samples/radian")
    samples = [path[0].copy()]
    for first, last in pairwise(path):
        count = max(80, int(np.ceil(np.max(np.abs(last - first)) * steps_per_radian)))
        t = np.linspace(0, 1, count + 1)[1:]
        blend = 10 * t**3 - 15 * t**4 + 6 * t**5
        samples.extend(first + fraction * (last - first) for fraction in blend)
    return np.asarray(samples)


def screen_solver_factory(factory, checker, stats, recover=None, rejected=None):
    """Screen every fresh Lula solver, not an unused one-off instance."""

    def create():
        solver = factory()
        original = solver.compute_inverse_kinematics

        def filtered(*args, **kwargs):
            solution, ok = original(*args, **kwargs)
            if ok:
                stats["queries"] += 1
                ok = bool(checker(solution))
                stats["rejected"] += int(not ok)
                if not ok and rejected is not None:
                    rejected(solution, *args, **kwargs)
                if not ok and recover is not None:
                    solution, ok = recover(solver, solution, *args, **kwargs)
                    ok = bool(ok and checker(solution))
                    stats["recovered"] = stats.get("recovered", 0) + int(ok)
            return solution, ok

        solver.compute_inverse_kinematics = filtered
        return solver

    return create


def make_fixture_reset(fixture: dict):
    def reset(task):
        if task.cfg.mode != "collect" or getattr(task, "_fixture_restored", False):
            raise ValueError("Fixture restoration is allowed only once, before the single collect attempt")
        paths = {str(obj.path) for obj in task.cfg.task.obj_paths}
        if paths != set(fixture["init_state"]["object_poses"]):
            raise ValueError("Fixture object paths do not match this task")
        if not task.cfg.task.get("replay_restore_orientation", False):
            raise ValueError("Fixed scene requires restoring the recorded object orientations")
        task.reset_with_init_state(copy.deepcopy(fixture["init_state"]))
        task._fixture_restored = True

    return reset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--config",
        choices=("level2_pipette_to_rack", "level4_three_pipettes_to_rack", "level4_four_pipettes_to_rack"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--runtime-margin", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--override", action="append", default=[], help="Explicit development-only Hydra override")
    parser.add_argument("--trace-arm", action="store_true", help="Read-only per-step arm command/tracking diagnostics")
    parser.add_argument(
        "--trace-robot-contacts", action="store_true", help="Enable contact reports only; no contact-property changes"
    )
    parser.add_argument("--trace-tool-contacts", action="store_true", help="Also report real pipette/rack contacts")
    parser.add_argument(
        "--sync-scene-cadence",
        action="store_true",
        help="Development: configure every PhysicsScene at the requested Hz",
    )
    parser.add_argument(
        "--screen-rack", action="store_true", help="Development-only IK sphere/box clearance screen; changes planning"
    )
    parser.add_argument(
        "--rack-screen-parts", action="store_true", help="Use each real collider's bounds, retaining gaps"
    )
    parser.add_argument(
        "--grasp-roll-deg", type=float, default=0.0, help="Development grip roll about the pipette axis"
    )
    parser.add_argument(
        "--optimize-clearance", action="store_true", help="Try bounded local collision-aware IK recovery"
    )
    parser.add_argument(
        "--trace-clearance-rejections", action="store_true", help="Record up to 32 distinct rejected poses"
    )
    parser.add_argument(
        "--rack-screen-oriented", action="store_true", help="Fit oriented bounds to every real collider mesh"
    )
    parser.add_argument(
        "--rack-screen-margin", type=float, default=0.005, help="Development planning gap, not success tolerance"
    )
    parser.add_argument(
        "--probe-rrt", action="store_true", help="Rack-only RRT reachability diagnostic; execute NO robot path"
    )
    parser.add_argument(
        "--execute-rrt-approach",
        action="store_true",
        help="Simulation-only open-gripper approach diagnostic; stop before closure, never task success",
    )
    parser.add_argument(
        "--probe-close-lift",
        action="store_true",
        help="After verified RRT arrival run original settle/close/lift only; no placement or task success",
    )
    parser.add_argument(
        "--continue-policy",
        action="store_true",
        help="Development full collect: verified RRT approach then original close/carry/place gates for every tool",
    )
    parser.add_argument(
        "--multistart-seat-ik",
        action="store_true",
        help="Development seated-pose multistart IK; original continuous motion and success gates retained",
    )
    parser.add_argument("--rrt-carry", action="store_true", help="Development held-tool/finger-aware global carry")
    parser.add_argument(
        "--screen-placement-geometry",
        action="store_true",
        help="Screen insertion/retreat with the measured open hand against the rack and other tools",
    )
    parser.add_argument(
        "--carry-screen-margin",
        type=float,
        default=0.002,
        help="Development carry-only planning buffer [0.002,0.02] m; never changes physical colliders",
    )
    parser.add_argument(
        "--full-approach-geometry", action="store_true", help="Plan with open fingers, bench and other tools"
    )
    parser.add_argument(
        "--measured-hand-cover", action="store_true", help="Experimental enabled physical-hand mesh sphere cover"
    )
    parser.add_argument(
        "--screen-rack-proposal", action="store_true", help="Try rack-only proposal but require full-scene screening"
    )
    parser.add_argument("--grasp-roll-candidates-deg", type=float, nargs="+", help="Ordered seat-aware grip rolls")
    parser.add_argument("--symmetric-seat-grasp", action="store_true", help="Also test local-axis opposing-jaw swaps")
    parser.add_argument(
        "--seat-heading-candidates-deg",
        type=float,
        nargs="+",
        help="Development motion-only release headings about rack vertical; observed goals/success checks unchanged",
    )
    parser.add_argument("--pregrasp-descent", type=float, default=0.0, help="Target-aware hover height in metres")
    parser.add_argument(
        "--shortcut-approach", action="store_true", help="Shorten approach only after full-geometry rescreening"
    )
    parser.add_argument(
        "--optimize-pose-path", action="store_true", help="Compare bounded native RRT paths to alternative IK endpoints"
    )
    parser.add_argument(
        "--approach-hold-compensation",
        action="store_true",
        help="Development bounded, geometry-screened endpoint bias; retains original physical arrival gate",
    )
    parser.add_argument(
        "--joint-steps-per-radian",
        type=float,
        default=400,
        help="Development-only joint interpolation density in [200,800]; lower is faster, default unchanged",
    )
    args = parser.parse_args()
    if args.seat_heading_candidates_deg is not None:
        headings = args.seat_heading_candidates_deg
        if (
            not args.execute_rrt_approach
            or not args.continue_policy
            or not args.multistart_seat_ik
            or args.grasp_roll_candidates_deg is None
            or not 1 <= len(headings) <= 9
            or headings[0] != 0
            or any(not np.isfinite(yaw) or not -20 <= yaw <= 20 for yaw in headings)
        ):
            parser.error(
                "Seat headings require full approach/continuation/multistart and 1..9 bounded angles starting at 0"
            )
    if not np.isfinite(args.carry_screen_margin) or not 0.002 <= args.carry_screen_margin <= 0.02:
        parser.error("Carry planning margin must be finite and in [0.002,0.02] m")
    if args.carry_screen_margin != 0.002 and not args.rrt_carry:
        parser.error("A custom carry margin requires --rrt-carry")
    if args.shortcut_approach and not args.full_approach_geometry:
        parser.error("Approach shortcut requires full approach geometry")
    if args.optimize_pose_path and not args.shortcut_approach:
        parser.error("Pose-path optimization requires the fully screened approach shortcut")
    if args.approach_hold_compensation and not args.full_approach_geometry:
        parser.error("Approach hold compensation requires full approach geometry")
    if not np.isfinite(args.joint_steps_per_radian) or not 200 <= args.joint_steps_per_radian <= 800:
        parser.error("Joint interpolation density must be finite and in [200,800] samples/radian")
    interpolate = partial(smooth_joint_path, steps_per_radian=args.joint_steps_per_radian)
    if args.symmetric_seat_grasp and args.grasp_roll_candidates_deg is None:
        parser.error("Symmetric seated grasp requires an explicit roll candidate list")
    if args.pregrasp_descent and (
        not args.full_approach_geometry
        or not np.isfinite(args.pregrasp_descent)
        or not 0.04 <= args.pregrasp_descent <= 0.15
    ):
        parser.error("Pregrasp descent requires full geometry and a finite hover height in [0.04,0.15] m")
    if args.grasp_roll_candidates_deg is not None and (
        not args.full_approach_geometry
        or len(args.grasp_roll_candidates_deg) > 9
        or not all(np.isfinite(r) and abs(r) <= 90 for r in args.grasp_roll_candidates_deg)
    ):
        parser.error("Seat-aware rolls require full geometry and 1..9 finite angles in [-90,90]")
    if args.screen_rack_proposal and not args.full_approach_geometry:
        parser.error("Rack proposal screening requires full approach geometry")
    if args.full_approach_geometry and not args.execute_rrt_approach:
        parser.error("Full approach geometry requires executed, contact-monitored RRT approach")
    if args.trace_tool_contacts and not args.trace_robot_contacts:
        parser.error("Tool contacts require --trace-robot-contacts")
    if args.screen_placement_geometry and not (
        args.rrt_carry and args.full_approach_geometry and args.measured_hand_cover and args.optimize_pose_path
    ):
        parser.error("Placement geometry screening requires full measured-hand approach and optimized RRT carry")
    if args.rrt_carry and not args.continue_policy:
        parser.error("RRT carry requires --continue-policy")
    if args.multistart_seat_ik and not args.continue_policy:
        parser.error("Multistart seated IK requires --continue-policy")
    if args.continue_policy and (not args.execute_rrt_approach or args.probe_close_lift):
        parser.error("Full-policy continuation requires executed approach and excludes --probe-close-lift")
    if args.probe_close_lift and not args.execute_rrt_approach:
        parser.error("Close/lift probe requires --execute-rrt-approach")
    if args.execute_rrt_approach and not (args.probe_rrt and args.trace_robot_contacts):
        parser.error("Executed approach requires --probe-rrt and --trace-robot-contacts")
    if args.probe_rrt and not args.screen_rack:
        parser.error("RRT feasibility probe requires --screen-rack geometry")
    if not np.isfinite(args.rack_screen_margin) or not 0 <= args.rack_screen_margin <= 0.03:
        parser.error("Rack planning margin must be finite and in [0,0.03] m")
    if args.optimize_clearance and not args.screen_rack:
        parser.error("Clearance optimization requires --screen-rack")
    if not np.isfinite(args.grasp_roll_deg) or abs(args.grasp_roll_deg) > 90:
        parser.error("Grasp roll must be finite and within +/-90 degrees")
    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name=args.config, overrides=args.override)
    OmegaConf.set_struct(cfg, False)
    cfg.mode = "collect"
    cfg.max_episodes = 1
    cfg.multi_run.run_dir = str(out / "run")
    cfg.task.pipette_enforce_runtime_joint_margin = args.runtime_margin == "enabled"
    OmegaConf.save(cfg, out / "attempt.yaml")
    main_args = [
        "--config-dir",
        os.path.relpath(out, ROOT),
        "--config-name",
        "attempt",
        "--headless",
        "--no-video",
        "--max-episodes",
        "1",
        "--max-attempts",
        "1",
        "--seed",
        str(args.seed),
        "--single-camera",
        "0",
        "--render-width",
        "256",
        "--render-height",
        "256",
        "--save-png-dir",
        str(out / "frames"),
        "--save-png-on-failure",
    ]
    os.environ["LABUTOPIA_PIPETTE_GRASP_DEBUG"] = "1"
    sys.argv = [str(ROOT / "experiments/main.py"), *main_args]
    # SimulationApp must be initialized before importing Isaac-dependent types.
    simulation = importlib.import_module("experiments.main")
    task_types = [
        importlib.import_module("tasks.pipette_rack_task").PipetteRackTask,
        importlib.import_module("tasks.pipette_rack_sequence_task").PipetteRackSequenceTask,
    ]
    controller_module = importlib.import_module("controllers.pipette_rack_controller")
    controller = controller_module.PipetteRackTaskController
    telemetry = importlib.import_module("experiments.support.grasp_telemetry")
    original_resets = {task_type: task_type.reset for task_type in task_types}
    original_setup = task_types[0].setup_objects
    original_step, own_step = controller.step, controller.__dict__.get("step")
    motion_class = controller_module.PipettePoseMotion
    original_motion = motion_class.forward
    original_recovery = controller_module.continuous_inverse_kinematics
    original_orientation = controller._pick_orientation
    original_plan = controller._plan_grasp_branch
    original_forward = controller._forward_active
    original_capture = controller._capture_grasp_offset
    original_close = simulation.simulation_app.close
    sources = [
        Path(__file__),
        ROOT / "experiments/main.py",
        Path(controller_module.__file__),
        Path(telemetry.__file__),
        ROOT / "robots/franka/franka.py",
        ROOT / "utils/gripper_contact.py",
        ROOT / "utils/franka_mimic_force.py",
        *[Path(importlib.import_module(task_type.__module__).__file__) for task_type in task_types],
    ]
    if args.sync_scene_cadence:
        sources.append(ROOT / "experiments/support/physics_scene_steps.py")
    if args.screen_rack:
        sources.append(ROOT / "controllers/pipette_clearance.py")
    if args.rrt_carry or args.full_approach_geometry:
        sources.append(ROOT / "experiments/pipette/pipette_carry_probe.py")
        sources.append(ROOT / "controllers/pipette_path_planning.py")
        sources.append(ROOT / "controllers/pipette_kinematics.py")
    with (out / "grip.jsonl").open("x") as stream:
        stream.write(
            json.dumps(
                {
                    "type": "metadata",
                    "argv": main_args,
                    "fixed_scene_initialization": True,
                    "read_only_step_observer": True,
                    "independent_sample": False,
                    "fixture": fixture,
                    "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
                    "runtime_margin": args.runtime_margin,
                    "config_overrides": args.override,
                    "trace_arm": args.trace_arm,
                    "development_sync_scene_cadence": args.sync_scene_cadence,
                    "robot_contact_reporting": args.trace_robot_contacts,
                    "development_all_phase_contact_guard": args.execute_rrt_approach,
                    "development_seat_heading_candidates_deg": args.seat_heading_candidates_deg,
                    "tool_contact_reporting": args.trace_tool_contacts,
                    "development_collision_screening": args.screen_rack,
                    "rack_screen_parts": args.rack_screen_parts,
                    "rack_screen_oriented": args.rack_screen_oriented,
                    "rack_screen_margin": args.rack_screen_margin,
                    "rrt_feasibility_only": args.probe_rrt and not args.execute_rrt_approach,
                    "development_open_gripper_approach": args.execute_rrt_approach,
                    "development_close_lift_probe": args.probe_close_lift,
                    "development_full_policy_continuation": args.continue_policy,
                    "development_multistart_seat_ik": args.multistart_seat_ik,
                    "development_rrt_carry": args.rrt_carry,
                    "development_placement_geometry_screen": args.screen_placement_geometry,
                    "development_carry_screen_margin": args.carry_screen_margin,
                    "development_full_approach_geometry": args.full_approach_geometry,
                    "development_measured_hand_cover": args.measured_hand_cover,
                    "development_screen_rack_proposal": args.screen_rack_proposal,
                    "development_grasp_roll_candidates_deg": args.grasp_roll_candidates_deg,
                    "development_symmetric_seat_grasp": args.symmetric_seat_grasp,
                    "development_pregrasp_descent": args.pregrasp_descent,
                    "development_shortcut_approach": args.shortcut_approach,
                    "development_optimize_pose_path": args.optimize_pose_path,
                    "development_approach_hold_compensation": args.approach_hold_compensation,
                    "development_joint_steps_per_radian": args.joint_steps_per_radian,
                    "development_clearance_optimization": args.optimize_clearance,
                    "trace_clearance_rejections": args.trace_clearance_rejections,
                    "development_grasp_roll_deg": args.grasp_roll_deg,
                    "source_sha256": {
                        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                    },
                }
            )
            + "\n"
        )
        last_terminal = False
        close_requested = False
        screening_stats = {"queries": 0, "rejected": 0}
        approach = {}
        external_contact = None
        selected_seat_headings = {}

        def placement_command_state(state):
            # Called ONLY for motion generation/capture, never for controller.step,
            # telemetry, stored init_state, _scene_state or success verification.
            heading = selected_seat_headings.get(state["object_path"], 0.0)
            if heading == 0:
                return state
            helper = importlib.import_module("experiments.pipette.pipette_carry_probe")
            return helper.seat_heading_command_state(state, heading)

        arm_properties_recorded = False
        physics_clock = {"ticks": 0, "last_dt_s": None, "minimum_dt_s": None, "maximum_dt_s": None}

        def emit(sample):
            nonlocal last_terminal
            last_terminal = sample["done"]
            stream.write(json.dumps(sample, allow_nan=False) + "\n")
            if last_terminal:
                stream.flush()

        def close(*close_args, **close_kwargs):
            nonlocal close_requested
            close_requested = True
            if args.screen_rack:
                stream.write(json.dumps({"type": "screening_stats", **screening_stats}) + "\n")
            stream.write(json.dumps({"type": "shutdown_requested", "last_step_terminal": last_terminal}) + "\n")
            stream.flush()
            return original_close(*close_args, **close_kwargs)

        def motion_forward(motion, *motion_args, **motion_kwargs):
            try:
                return original_motion(motion, *motion_args, **motion_kwargs)
            except ValueError:
                # Diagnostic only: observe the failing solve without requesting
                # another solution or changing the exception/action flow.
                record = {
                    "type": "motion_failure",
                    "args": motion_args,
                    "kwargs": motion_kwargs,
                    "seed": motion.seed,
                    "goal": motion.goal,
                    "step": motion.step,
                    "duration": getattr(motion, "duration", None),
                    "joint_margin": motion.joint_margin,
                    "joint_bounds": motion.joint_bounds,
                    "actual_joints": motion.rmp.articulation_rmp.get_active_joints_subset().get_joint_positions(),
                }
                stream.write(json.dumps(record, default=lambda value: value.tolist()) + "\n")
                stream.flush()
                raise

        guarded_step = (
            with_contact_guard(original_step, lambda: external_contact, controller_module.Phase.FINISHED)
            if args.execute_rrt_approach
            else original_step
        )
        observed_grip_step = telemetry.observed_step(guarded_step, emit)
        checker = None
        screened_flow = original_solver_factory = None

        def configure_clearance(active, state):
            nonlocal checker, screened_flow, original_solver_factory
            if checker is not None:
                return
            clearance_module = importlib.import_module("controllers.pipette_clearance")
            center, size = state["target_position"], state["target_size"]
            boxes = [[center - size / 2, center + size / 2]]
            box_paths = [state["target_path"]]
            box_frames = None
            if args.rack_screen_parts or args.rack_screen_oriented:
                module = importlib.import_module("tasks.pipette_rack_task")
                stage = simulation.omni.usd.get_context().get_stage()
                cache = module.UsdGeom.BBoxCache(0, ["default"])
                boxes = []
                box_paths = []
                box_frames = [] if args.rack_screen_oriented else None
                for prim in module.Usd.PrimRange(stage.GetPrimAtPath(state["target_path"])):
                    if not prim.HasAPI(module.UsdPhysics.CollisionAPI):
                        continue
                    if not module.UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                        continue
                    if args.rack_screen_oriented:
                        if not prim.IsA(module.UsdGeom.Mesh):
                            raise ValueError("Oriented rack screening requires the actual collider mesh vertices")
                        transform = module.UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                        points = np.asarray(
                            [
                                transform.Transform(module.Gf.Vec3d(*point))
                                for point in module.UsdGeom.Mesh(prim).GetPointsAttr().Get()
                            ]
                        )
                        bounds, frame = clearance_module.oriented_mesh_bounds(points)
                        boxes.append(bounds)
                        box_frames.append(frame)
                    else:
                        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                        boxes.append([np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())])
                    box_paths.append(str(prim.GetPath()))
            screened_flow = active.rmp_controller.rmp_flow
            original_solver_factory = screened_flow.get_kinematics_solver
            checker = clearance_module.RackArmClearance.from_descriptor(
                original_solver_factory(),
                Path(active.rmp_controller.rmp_flow_config["robot_description_path"]),
                boxes,
                margin=args.rack_screen_margin,
                box_frames=box_frames,
            )

            def filtered_recovery(*recovery_args, **recovery_kwargs):
                solution, ok = original_recovery(*recovery_args, **recovery_kwargs)
                return solution, bool(ok and checker(solution))

            def recover(solver, seed, frame, position, quaternion, **ignored):
                bounds = np.asarray(active.robot._articulation_view.get_dof_limits())[0, :7]
                return clearance_module.recover_clearance_ik(
                    solver,
                    checker,
                    frame,
                    position,
                    quaternion,
                    seed,
                    bounds,
                    joint_margin=float(cfg.task.pipette_preflight_joint_margin),
                )

            rejected_poses = set()

            def rejected(solution, frame, position, quaternion, **ignored):
                key = tuple(np.round(np.r_[position, quaternion], 2))
                if key in rejected_poses or len(rejected_poses) >= 32:
                    return
                rejected_poses.add(key)
                contact = checker.limiting_contact(solution)
                stream.write(
                    json.dumps(
                        {
                            "type": "clearance_rejection",
                            "target": np.asarray(position).tolist(),
                            "orientation": np.asarray(quaternion).tolist(),
                            "joints": np.asarray(solution).tolist(),
                            "limiting": contact,
                            "obstacle": box_paths[contact["box"]],
                        }
                    )
                    + "\n"
                )
                stream.flush()

            screened_flow.get_kinematics_solver = screen_solver_factory(
                original_solver_factory,
                checker,
                screening_stats,
                recover if args.optimize_clearance else None,
                rejected if args.trace_clearance_rejections else None,
            )
            controller_module.continuous_inverse_kinematics = filtered_recovery
            stream.write(
                json.dumps(
                    {
                        "type": "clearance_screen",
                        "boxes": checker.boxes.tolist(),
                        "box_paths": box_paths,
                        "box_frames": None if checker.box_frames is None else checker.box_frames.tolist(),
                        "frames": list(checker.frames),
                        "margin": checker.margin,
                    }
                )
                + "\n"
            )

        def observed_arm_step(active, state):
            nonlocal arm_properties_recorded
            # Capture the last applied command before issuing the next one.
            # These diagnostics never call an IK solver or apply an action.
            if args.screen_rack:
                configure_clearance(active, state)
            if not arm_properties_recorded:
                stream.write(
                    json.dumps(
                        read_arm_drive_properties(active.robot), default=lambda value: value.tolist(), allow_nan=False
                    )
                    + "\n"
                )
                arm_properties_recorded = True
            applied = active.robot.get_articulation_controller().get_applied_action()
            measured = state["joint_positions"].copy()
            dynamics = read_arm_dynamics(active.robot, applied)
            result = observed_grip_step(active, state)
            motion = getattr(active.active_controller, "_cspace_controller", None)
            requested = result[0]
            record = {
                "type": "arm_sample",
                "phase": active.current_phase.value,
                "event": getattr(active.active_controller, "_event", None),
                "measured": measured,
                **dynamics,
                "physics_clock": physics_clock.copy(),
                "requested_positions": None if requested is None else requested.joint_positions,
                "requested_indices": None if requested is None else requested.joint_indices,
                "seed": getattr(motion, "seed", None),
                "tracking_bias": getattr(motion, "tracking_bias", None),
                "goal": getattr(motion, "goal", None),
                "step": getattr(motion, "step", None),
                "duration": getattr(motion, "duration", None),
            }
            stream.write(json.dumps(record, default=lambda value: value.tolist(), allow_nan=False) + "\n")
            return result

        def setup_with_contact_reports(task):
            original_setup(task)
            module = importlib.import_module("tasks.pipette_rack_task")
            if args.sync_scene_cadence:
                cadence = importlib.import_module("experiments.support.physics_scene_steps")
                scene_records = cadence.synchronize_physics_scene_steps(task.stage, 60 * int(cfg.physics_substeps))
                stream.write(json.dumps({"type": "physics_scene_cadence", "scenes": scene_records}) + "\n")
            if args.trace_arm:

                def clock_step(dt):
                    physics_clock["ticks"] += 1
                    physics_clock["last_dt_s"] = float(dt)
                    physics_clock["minimum_dt_s"] = min(physics_clock["minimum_dt_s"] or float(dt), float(dt))
                    physics_clock["maximum_dt_s"] = max(physics_clock["maximum_dt_s"] or float(dt), float(dt))

                task._fixture_physics_clock_subscription = (
                    simulation.omni.physx.get_physx_interface().subscribe_physics_step_events(clock_step)
                )
            if not args.trace_robot_contacts:
                return
            robot_path = task.robot.prim_path_str
            for prim in module.Usd.PrimRange(task.stage.GetPrimAtPath(robot_path)):
                if prim.HasAPI(module.UsdPhysics.RigidBodyAPI):
                    module.PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            tool_paths = [
                str(obj.path) for obj in cfg.task.obj_paths if str(obj.path) != str(cfg.task.obj_paths[1].path)
            ]
            if args.trace_tool_contacts:
                for prim_path in tool_paths:
                    for prim in module.Usd.PrimRange(task.stage.GetPrimAtPath(prim_path)):
                        if prim.HasAPI(module.UsdPhysics.RigidBodyAPI):
                            module.PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)

            def report(headers, contacts):
                nonlocal external_contact
                if not args.execute_rrt_approach and getattr(task, "frame_idx", 0) % 20:
                    return
                for header in headers:
                    paths = [
                        str(module.PhysicsSchemaTools.intToSdfPath(value))
                        for value in (header.collider0, header.collider1)
                    ]
                    robot_contact = any(p.startswith(robot_path + "/") for p in paths)
                    tool_contact = args.trace_tool_contacts and any(
                        p.startswith(root + "/") for p in paths for root in tool_paths
                    )
                    if not robot_contact and not tool_contact:
                        continue
                    for contact in contacts[
                        header.contact_data_offset : header.contact_data_offset + header.num_contact_data
                    ]:
                        impulse = np.array([contact.impulse.x, contact.impulse.y, contact.impulse.z], dtype=float)
                        moving_external = sum(p.startswith(robot_path + "/") for p in paths) == 1 and not any(
                            p.startswith(robot_path + "/panda_link0/") for p in paths
                        )
                        allowed_grasp = approach.get("grip_phase", False) and (
                            any(p.startswith(approach["object_path"] + "/") for p in paths)
                            and any(
                                p.startswith((robot_path + "/panda_leftfinger/", robot_path + "/panda_rightfinger/"))
                                for p in paths
                            )
                        )
                        if (
                            external_contact is None
                            and approach
                            and moving_external
                            and not allowed_grasp
                            and np.linalg.norm(impulse) > 0.001
                        ):
                            external_contact = {
                                "paths": paths,
                                "impulse_norm": float(np.linalg.norm(impulse)),
                                "frame": int(getattr(task, "frame_idx", 0)),
                            }
                        stream.write(
                            json.dumps(
                                {
                                    "type": "robot_contact" if robot_contact else "object_contact",
                                    "frame": task.frame_idx,
                                    "paths": paths,
                                    "separation": float(contact.separation),
                                    "point_world": [
                                        float(contact.position.x),
                                        float(contact.position.y),
                                        float(contact.position.z),
                                    ],
                                    "normal_world": [
                                        float(contact.normal.x),
                                        float(contact.normal.y),
                                        float(contact.normal.z),
                                    ],
                                    "impulse": [
                                        float(contact.impulse.x),
                                        float(contact.impulse.y),
                                        float(contact.impulse.z),
                                    ],
                                }
                            )
                            + "\n"
                        )

            task._fixture_robot_contact_subscription = (
                module.get_physx_simulation_interface().subscribe_contact_report_events(report)
            )

        def rolled_orientation(active, *, settled_state=None):
            orientation = original_orientation(active, settled_state=settled_state)
            if active._planned_pick_orientation is not None:
                return orientation  # already rolled and locked by preflight
            state = active.state if settled_state is None else settled_state
            axis = np.asarray(state["object_axis"], dtype=float)
            axis = axis / np.linalg.norm(axis)
            roll = Rotation.from_rotvec(axis * np.radians(args.grasp_roll_deg))
            return (roll * Rotation.from_quat(orientation[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]

        def probe_rrt(active, state, tcp_local):
            module = importlib.import_module("isaacsim.robot_motion.motion_generation")
            geometry = importlib.import_module("controllers.pipette_clearance")
            planner_config = module.interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
            for key in ("robot_description_path", "urdf_path", "end_effector_frame_name"):
                planner_config[key] = active.rmp_controller.rmp_flow_config[key]
            path_checker = checker
            if args.full_approach_geometry:
                helper = importlib.import_module("experiments.pipette.pipette_carry_probe")
                usd = importlib.import_module("tasks.pipette_rack_task")
                descriptor, path_checker, details = helper.open_hand_planning_scene(
                    active,
                    state,
                    checker,
                    usd,
                    simulation.omni.usd.get_context().get_stage(),
                    out,
                    measured_hand=args.measured_hand_cover,
                    include_target=bool(args.pregrasp_descent),
                )
                planner_config["robot_description_path"] = str(descriptor)
                stream.write(json.dumps(details) + "\n")
            planner_class = importlib.import_module("isaacsim.robot_motion.motion_generation.lula.path_planners").RRT
            planner = planner_class(**planner_config)
            planner.set_robot_base_pose(*active.robot.get_world_pose())
            planner.set_random_seed(args.seed)
            planner.set_max_iterations(4000 if args.full_approach_geometry else 2000)
            planner.set_param("step_size", 0.03)
            frames = path_checker.box_frames
            if frames is None:
                frames = np.tile(np.eye(4), (len(path_checker.boxes), 1, 1))
            obstacles = [
                geometry.PlanningBox(box, frame, path_checker.margin) for box, frame in zip(path_checker.boxes, frames)
            ]
            for obstacle in obstacles:
                if not planner.add_cuboid(obstacle, static=True):
                    raise ValueError("Could not register the in-memory RRT rack obstacle")
            planner.update_world()
            orientation = active._pick_orientation(settled_state=state)
            if args.grasp_roll_candidates_deg is not None:
                orientation = helper.choose_seat_grasp(
                    active,
                    state,
                    orientation,
                    tcp_local,
                    args.grasp_roll_candidates_deg,
                    args.grasp_roll_deg,
                    args.seed,
                    lambda record: stream.write(json.dumps(record) + "\n"),
                    symmetric=args.symmetric_seat_grasp,
                    pregrasp_checker=path_checker if args.pregrasp_descent else None,
                    pregrasp_height=args.pregrasp_descent,
                    seat_headings=args.seat_heading_candidates_deg or (0.0,),
                    on_heading_selected=lambda heading: selected_seat_headings.__setitem__(
                        state["object_path"], heading
                    ),
                )
            target = np.asarray(state["object_grasp_position"]) + np.array(
                [0, 0, float(cfg.task.pipette_grasp_height_offset)]
            )
            target = target - Rotation.from_quat(orientation[[1, 2, 3, 0]]).apply(tcp_local)
            grasp_target = target.copy()
            target = target + np.array([0, 0, args.pregrasp_descent])
            planner.set_end_effector_target(target, orientation)
            start = active.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
            path = None
            if args.screen_rack_proposal:
                # A cheap proposal is never executed on its original rack-only
                # verdict. Actual open fingers, bench and other tools must all
                # pass the same full-scene dense check as a fresh full plan.
                proposal_config = planner_config.copy()
                proposal_config["robot_description_path"] = active.rmp_controller.rmp_flow_config[
                    "robot_description_path"
                ]
                proposal = planner_class(**proposal_config)
                proposal.set_robot_base_pose(*active.robot.get_world_pose())
                proposal.set_random_seed(args.seed)
                proposal.set_max_iterations(2000)
                proposal.set_param("step_size", 0.03)
                rack_frames = checker.box_frames
                if rack_frames is None:
                    rack_frames = np.tile(np.eye(4), (len(checker.boxes), 1, 1))
                for box, frame in zip(checker.boxes, rack_frames):
                    if not proposal.add_cuboid(geometry.PlanningBox(box, frame, checker.margin), static=True):
                        raise ValueError("Could not register proposal obstacle")
                proposal.update_world()
                proposal.set_end_effector_target(target, orientation)
                candidate = proposal.compute_path(np.asarray(start), np.empty(0))
                verdict = {"accepted": False, "path_found": candidate is not None}
                if candidate is not None:
                    verdict.update(
                        helper.screen_joint_path(
                            candidate,
                            path_checker,
                            np.asarray(active.robot._articulation_view.get_dof_limits())[0, :7],
                            joint_margin=float(cfg.task.pipette_preflight_joint_margin),
                        )
                    )
                    if verdict["accepted"]:
                        path = candidate
                    verdict["limiting_contact"] = path_checker.limiting_contact(verdict["limiting_joints"])
                stream.write(json.dumps({"type": "full_scene_proposal_screen", **verdict}) + "\n")
            if path is None:
                path = planner.compute_path(np.asarray(start), np.empty(0))
            original_rrt_path = None if path is None else np.asarray(path).tolist()
            found = path is not None and len(path) > 0
            shortcut_record = {}
            if found and args.shortcut_approach:
                path, shortcut_record = helper.prepare_approach_path(
                    path,
                    path_checker,
                    np.asarray(active.robot._articulation_view.get_dof_limits())[0, :7],
                    interpolate,
                    joint_margin=float(cfg.task.pipette_preflight_joint_margin),
                )
            if args.optimize_pose_path:
                path, details = helper.improve_pose_path(
                    planner,
                    path_checker,
                    planner_config["end_effector_frame_name"],
                    target,
                    orientation,
                    start,
                    path,
                    np.asarray(active.robot._articulation_view.get_dof_limits())[0, :7],
                    interpolate,
                    args.seed,
                    joint_margin=float(cfg.task.pipette_preflight_joint_margin),
                )
                stream.write(
                    json.dumps(
                        {"type": "approach_pose_path_optimization", "object_path": state["object_path"], **details}
                    )
                    + "\n"
                )
                found = path is not None and len(path) > 0
                if found:
                    path, shortcut_record = helper.prepare_approach_path(
                        path,
                        path_checker,
                        np.asarray(active.robot._articulation_view.get_dof_limits())[0, :7],
                        interpolate,
                        joint_margin=float(cfg.task.pipette_preflight_joint_margin),
                    )
            record = {
                "type": "rrt_feasibility",
                "rack_only_obstacles": not args.full_approach_geometry,
                "path_executed": False,
                "path_found": found,
                "target": target.tolist(),
                "grasp_target": grasp_target.tolist(),
                "orientation": orientation.tolist(),
                "path": None if path is None else np.asarray(path).tolist(),
                "original_rrt_path": original_rrt_path,
                **shortcut_record,
            }
            if found:
                limits = np.asarray(active.robot._articulation_view.get_dof_limits())[0, :7]
                record["minimum_joint_margin"] = float(np.min(np.minimum(path - limits[:, 0], limits[:, 1] - path)))
                record["minimum_vertex_rack_gap"] = min(path_checker.minimum_clearance(q) for q in path)
                dense = [np.asarray(path[0])]
                for first, last in pairwise(path):
                    steps = max(1, int(np.ceil(np.max(np.abs(last - first)) / 0.01)))
                    dense.extend(first + t * (last - first) for t in np.linspace(0, 1, steps + 1)[1:])
                record["dense_sample_count"] = len(dense)
                record["minimum_dense_rack_gap"] = min(path_checker.minimum_clearance(q) for q in dense)
                record["gap_includes_bench_other_tools_fingers"] = args.full_approach_geometry
                usd = importlib.import_module("tasks.pipette_rack_task")
                stage = simulation.omni.usd.get_context().get_stage()
                cache = usd.UsdGeom.BBoxCache(0, ["default"])
                extra_boxes, extra_paths = [], []
                candidates = [str(obj.path) for obj in cfg.task.obj_paths if str(obj.path) != state["target_path"]]
                candidates.append("/World/Cube")  # the actual bench, not an invented floor
                for candidate in candidates:
                    prim = stage.GetPrimAtPath(candidate)
                    if not prim.IsValid():
                        raise ValueError(f"Missing physical scene obstacle: {candidate}")
                    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    extra_boxes.append([np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())])
                    extra_paths.append(candidate)
                moving_spheres = {
                    frame: [{"center": center, "radius": radius} for center, radius in zip(centers, radii)]
                    for frame, (centers, radii) in checker.frames.items()
                    if frame != "panda_link0"
                }
                extra_checker = geometry.RackArmClearance(checker.solver, moving_spheres, extra_boxes, margin=0)
                contacts = [extra_checker.limiting_contact(q) for q in dense]
                limiting = min(contacts, key=lambda contact: contact["clearance"])
                record["moving_arm_scene_obstacles"] = extra_paths
                record["minimum_moving_arm_scene_gap"] = limiting["clearance"]
                record["limiting_scene_contact"] = {**limiting, "obstacle": extra_paths[limiting["box"]]}
                actual_position, actual_matrix = checker.solver.compute_forward_kinematics(
                    planner_config["end_effector_frame_name"], path[-1]
                )
                record["endpoint_position_error"] = float(
                    np.linalg.norm(np.asarray(actual_position).reshape(3) - target)
                )
                record["endpoint_orientation_error"] = float(
                    (
                        Rotation.from_quat(orientation[[1, 2, 3, 0]]) * Rotation.from_matrix(actual_matrix).inv()
                    ).magnitude()
                )
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if args.execute_rrt_approach:
                if not found or record["minimum_dense_rack_gap"] < checker.margin:
                    raise ValueError("No rack-clear RRT approach available for physical diagnostic")
                if record["minimum_joint_margin"] < float(cfg.task.pipette_preflight_joint_margin):
                    raise ValueError("RRT approach violates the configured physical joint margin")
                approach["checker"] = path_checker
                return record
            raise ValueError(f"RRT feasibility diagnostic finished: path_found={found}; no robot path executed")

        def approach_forward(active, state):
            nonlocal external_contact
            # Preserve normal scene settle and collect action caching. This
            # Optional close/lift retains original physical gates, but NEVER
            # proceeds to placement or reports whole-task success.
            if args.continue_policy and approach.get("grip_phase", False):
                if state["object_path"] == approach["object_path"]:
                    if (
                        args.rrt_carry
                        and active.current_phase == controller_module.Phase.PLACING
                        and active._transit_phase < 2
                    ):
                        return carry_forward(active, state)
                    return original_forward(active, state)
                # The sequence reset its controller after verified placement;
                # retain the physical scene and plan only the next empty-hand approach.
                approach.clear()
                external_contact = None
            if active._preflight_wait_steps < int(cfg.task.pipette_scene_settle_steps):
                return original_forward(active, state)
            pick = active.pick_controller

            def finish(reason, reached=False):
                stream.write(
                    json.dumps(
                        {
                            "type": "rrt_approach_result",
                            "reached": reached,
                            "task_success": False,
                            "reason": reason,
                            "steps": approach.get("steps", 0),
                            "path_index": approach.get("index", 0),
                            "external_contact": external_contact,
                            "close_lift_probe": args.probe_close_lift,
                        }
                    )
                    + "\n"
                )
                stream.flush()
                pick._phase_failure_reason = "Open-gripper diagnostic stopped: " + reason
                pick._is_done = True
                action = pick._null_action(len(state["joint_positions"]))
                return action, pick._build_record_array(
                    action, state["joint_positions"], gripper_state=pick._current_gripper_state
                )

            if approach.get("grip_phase", False):
                approach["grip_steps"] += 1
                if external_contact is not None:
                    return finish("Unexpected external contact during close/lift")
                if pick._event >= 6:
                    return finish(
                        "Original stable-close and lift arrival gates passed; placement untested", reached=True
                    )
                if approach["grip_steps"] > 1500:
                    return finish("Bounded 1500-step close/lift timeout")
                return original_forward(active, state)
            pick._open_gripper()
            joints = np.asarray(active.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions())
            position, matrix = active.rmp_controller.rmp_flow.get_end_effector_pose(joints)
            objects = state.get("pipettes", [state])
            if not approach:
                tcp_local = Rotation.from_matrix(matrix).inv().apply(state["gripper_position"] - position)
                try:
                    plan = probe_rrt(active, state, tcp_local)
                except ValueError as error:
                    return finish(str(error))
                approach.update(
                    plan=plan,
                    commands=interpolate(plan["path"]),
                    index=0,
                    steps=0,
                    stable=0,
                    endpoint_hold_steps=0,
                    endpoint_bias=np.zeros(7),
                    objects={
                        obj["object_path"]: (
                            np.asarray(obj["object_grasp_position"]).copy(),
                            Rotation.from_quat(obj["object_quaternion"]),
                        )
                        for obj in objects
                    },
                )
            approach["steps"] += 1
            if external_contact is not None:
                return finish("Moving robot made external contact exceeding 0.001 N s")
            for obj in objects:
                old_position, old_rotation = approach["objects"][obj["object_path"]]
                moved = float(np.linalg.norm(np.asarray(obj["object_grasp_position"]) - old_position))
                rotated = float((Rotation.from_quat(obj["object_quaternion"]) * old_rotation.inv()).magnitude())
                if moved > 0.001 or rotated > np.radians(0.5):
                    return finish(
                        f"Unclosed object moved: {obj['object_path']}, {moved:.6f} m, {np.degrees(rotated):.4f} deg"
                    )
            commands = approach["commands"]
            index = approach["index"]
            error = float(np.max(np.abs(joints - commands[index])))
            if error < 0.03 and index < len(commands) - 1:
                approach["index"] += 1
            plan = approach["plan"]
            position_error = float(np.linalg.norm(np.asarray(position) - plan["target"]))
            angle_error = float(
                (
                    Rotation.from_matrix(matrix)
                    * Rotation.from_quat(np.asarray(plan["orientation"])[[1, 2, 3, 0]]).inv()
                ).magnitude()
            )
            reached = index == len(commands) - 1 and position_error < 0.002 and angle_error < np.radians(2)
            approach["stable"] = approach["stable"] + 1 if reached else 0
            command = commands[approach["index"]].copy()
            if args.approach_hold_compensation and index == len(commands) - 1 and "descent" not in approach:
                approach["endpoint_hold_steps"] += 1
                if approach["endpoint_hold_steps"] >= 5:
                    helper = importlib.import_module("experiments.pipette.pipette_carry_probe")
                    try:
                        command, approach["endpoint_bias"], verdict = helper.screened_hold_compensation(
                            commands[-1],
                            joints,
                            approach["endpoint_bias"],
                            approach["checker"],
                            np.asarray(active.robot._articulation_view.get_dof_limits())[0, :7],
                            joint_margin=float(cfg.task.pipette_preflight_joint_margin),
                        )
                    except ValueError as exc:
                        return finish(str(exc))
                    approach["endpoint_bias_screen"] = verdict
            stream.write(
                json.dumps(
                    {
                        "type": "rrt_approach_sample",
                        "step": approach["steps"],
                        "index": index,
                        "joint_tracking_error": error,
                        "position_error": position_error,
                        "orientation_error": angle_error,
                        "actual_joints": joints.tolist(),
                        "endpoint_bias": approach["endpoint_bias"].tolist(),
                        "endpoint_bias_screen": approach.get("endpoint_bias_screen"),
                    }
                )
                + "\n"
            )
            if approach["stable"] >= 10 or "descent" in approach:
                if args.pregrasp_descent and "descent" not in approach:
                    helper = importlib.import_module("experiments.pipette.pipette_carry_probe")
                    tcp_local = Rotation.from_matrix(matrix).inv().apply(state["gripper_position"] - position)
                    motion = motion_class(
                        active.rmp_controller,
                        joints.copy(),
                        tcp_local,
                        cartesian_first=True,
                        transfer_time_scale=2.0,
                        local_ik_recovery=bool(cfg.task.get("pipette_local_ik_recovery", False)),
                        **active._motion_constraints(),
                    )
                    approach["descent"] = helper.PregraspDescent(
                        motion, plan["grasp_target"], plan["orientation"], tcp_local
                    )
                    stream.write(
                        json.dumps(
                            {
                                "type": "pregrasp_hover_arrival",
                                "position_error": position_error,
                                "orientation_error": angle_error,
                                "physical_arrival_frames": 10,
                            }
                        )
                        + "\n"
                    )
                if "descent" in approach:
                    try:
                        action, descent_record = approach["descent"].forward(position, matrix)
                    except ValueError as exc:
                        return finish(str(exc))
                    stream.write(json.dumps(descent_record) + "\n")
                    if not descent_record["reached"]:
                        return action, pick._build_record_array(
                            action, state["joint_positions"], gripper_state=pick._current_gripper_state
                        )
                    position_error = descent_record["position_error"]
                    angle_error = descent_record["orientation_error"]
                if args.probe_close_lift or args.continue_policy:
                    # Earlier approach phases have now physically reached a
                    # stricter (2 mm / 2 deg, 10-frame) arrival gate. Continue
                    # with original settle, stable closure, and lift checks.
                    active._planned_pick_orientation = np.asarray(plan["orientation"])
                    tcp_local = Rotation.from_matrix(matrix).inv().apply(state["gripper_position"] - position)
                    pick._cspace_controller = motion_class(
                        active.rmp_controller,
                        joints.copy(),
                        tcp_local,
                        cartesian_first=True,
                        local_ik_recovery=bool(cfg.task.get("pipette_local_ik_recovery", False)),
                        **active._motion_constraints(),
                    )
                    target = np.asarray(state["object_grasp_position"]).copy()
                    target[2] += float(cfg.task.pipette_grasp_height_offset)
                    pick._last_target = target.copy()
                    pick._locked_picking_position = target.copy()
                    pick._event, pick._t, pick._start = 3, 0.0, False
                    approach.update(grip_phase=True, grip_steps=0, object_path=state["object_path"])
                    stream.write(
                        json.dumps(
                            {
                                "type": "rrt_approach_handoff",
                                "position_error": position_error,
                                "orientation_error": angle_error,
                                "physical_arrival_frames": 10,
                            }
                        )
                        + "\n"
                    )
                    return original_forward(active, state)
                return finish("Reached grasp with open fingers; closure and carry untested", reached=True)
            approach_budget = max(2400, len(commands) + 600) if args.pregrasp_descent else 2400
            if approach["steps"] >= approach_budget:
                return finish(f"Bounded {approach_budget}-step approach timeout")
            action = controller_module.ArticulationAction(joint_positions=command, joint_indices=np.arange(7))
            return action, pick._build_record_array(
                action, state["joint_positions"], gripper_state=pick._current_gripper_state
            )

        def carry_forward(active, state):
            nonlocal external_contact

            def diagnostic(record):
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()

            try:
                if "carry" not in approach:
                    external_contact = None
                    carry_class = importlib.import_module("experiments.pipette.pipette_carry_probe").CarryRRTProbe
                    usd = importlib.import_module("tasks.pipette_rack_task")
                    approach["carry"] = carry_class(
                        active,
                        state,
                        checker,
                        usd,
                        simulation.omni.usd.get_context().get_stage(),
                        out,
                        args.seed,
                        diagnostic,
                        interpolate,
                        measured_hand=args.measured_hand_cover,
                        optimize_pose=args.optimize_pose_path,
                        carry_margin=args.carry_screen_margin,
                        placement_robot_frames=approach["checker"].frames if args.screen_placement_geometry else None,
                    )
                if external_contact is not None:
                    raise ValueError(f"Unexpected physical contact during carry: {external_contact}")
                carry = approach["carry"]
                action = carry.forward(active, state)
                if carry.done:
                    active._transit_phase = 2
                    joints = active.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
                    active.place_controller._cspace_controller = motion_class(
                        active.rmp_controller,
                        joints.copy(),
                        carry.tcp_local,
                        cartesian_first=True,
                        transfer_time_scale=float(cfg.task.pipette_transfer_time_scale),
                        local_ik_recovery=bool(cfg.task.get("pipette_local_ik_recovery", False)),
                        **active._motion_constraints(),
                    )
                return action, active.place_controller._build_record_array(
                    action, state["joint_positions"], gripper_state=active.place_controller._current_gripper_state
                )
            except ValueError as error:
                diagnostic({"type": "carry_rrt_failure", "reason": str(error)})
                return active._placement_failure(state, str(error))

        def capture_with_multistart(active, state):
            state = placement_command_state(state)
            flow = active.rmp_controller.rmp_flow
            factory = flow.get_kinematics_solver
            in_capture = True
            bounds = np.asarray(active.robot._articulation_view.get_dof_limits())[0, :7]
            margin = float(cfg.task.pipette_preflight_joint_margin)

            def create():
                solver = factory()
                solve = solver.compute_inverse_kinematics

                def multistart(frame, position, orientation, **kwargs):
                    solution, ok = solve(frame, position, orientation, **kwargs)
                    initial_margin = float(np.min(np.minimum(solution - bounds[:, 0], bounds[:, 1] - solution)))
                    if (ok and initial_margin >= margin) or not in_capture:
                        return solution, ok
                    current = np.asarray(kwargs["warm_start"])
                    seeds = [
                        np.array([shoulder, -0.7, -shoulder, -2.2, wrist, 2.4, roll])
                        for shoulder, wrist, roll in product((-1.5, 0.0, 1.5), (-2.0, 0.0, 2.0), (-2.0, 0.0, 2.0))
                    ]
                    seeds.extend(
                        np.random.default_rng(args.seed).uniform(
                            bounds[:, 0] + margin, bounds[:, 1] - margin, size=(64, 7)
                        )
                    )
                    records, candidates = [], []
                    for seed in seeds:
                        candidate, converged = solve(frame, position, orientation, **{**kwargs, "warm_start": seed})
                        candidate = np.asarray(candidate)
                        gap = float(np.min(np.minimum(candidate - bounds[:, 0], bounds[:, 1] - candidate)))
                        records.append(
                            {
                                "converged_and_rack_clear": bool(converged),
                                "joint_margin": gap,
                                "joints": candidate.tolist(),
                            }
                        )
                        if converged and gap >= margin:
                            candidates.append(candidate.copy())
                    selected = min(candidates, key=lambda q: np.linalg.norm(q - current)) if candidates else None
                    stream.write(
                        json.dumps(
                            {
                                "type": "seat_multistart",
                                "target": np.asarray(position).tolist(),
                                "orientation": np.asarray(orientation).tolist(),
                                "current": current.tolist(),
                                "attempts": records,
                                "feasible_count": len(candidates),
                                "selected": None if selected is None else selected.tolist(),
                                "continuous_carry_validated": False,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    stream.flush()
                    return (selected, True) if selected is not None else (solution, False)

                solver.compute_inverse_kinematics = multistart
                return solver

            flow.get_kinematics_solver = create
            try:
                return original_capture(active, state)
            finally:
                in_capture = False  # Retained motion solvers must NOT branch-hop during transport.
                flow.get_kinematics_solver = factory

        for task_type in task_types:
            task_type.reset = make_fixture_reset(fixture)
        if args.trace_robot_contacts or args.trace_arm or args.sync_scene_cadence:
            task_types[0].setup_objects = setup_with_contact_reports
        if args.grasp_roll_deg:
            controller._pick_orientation = rolled_orientation
        if args.probe_rrt:
            controller._plan_grasp_branch = probe_rrt
        if args.execute_rrt_approach:

            def forward_with_command_heading(active, state):
                return approach_forward(active, placement_command_state(state))

            controller._forward_active = forward_with_command_heading
        if args.multistart_seat_ik:
            controller._capture_grasp_offset = capture_with_multistart
        controller.step = observed_arm_step if args.trace_arm or args.screen_rack else observed_grip_step
        motion_class.forward = motion_forward
        simulation.simulation_app.close = close
        try:
            simulation.main()
        finally:
            for task_type, original in original_resets.items():
                task_type.reset = original
            if own_step is None:
                del controller.step
            else:
                controller.step = own_step
            simulation.simulation_app.close = original_close
            motion_class.forward = original_motion
            task_types[0].setup_objects = original_setup
            controller_module.continuous_inverse_kinematics = original_recovery
            controller._pick_orientation = original_orientation
            controller._plan_grasp_branch = original_plan
            controller._forward_active = original_forward
            controller._capture_grasp_offset = original_capture
            if screened_flow is not None:
                screened_flow.get_kinematics_solver = original_solver_factory
            if not close_requested:
                close()  # Flush diagnostics and release Isaac even on probe/API errors.


if __name__ == "__main__":
    main()
