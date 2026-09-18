"""Run the real entry point with read-only telemetry and optional conical-only phases.

CONICAL_ONLY=1 is an ISOLATED COMPONENT test, never a full Level-4 score.
The production scene, physics, randomization, cameras and atomic actions stay intact.
"""

import json
import os
from pathlib import Path

# SimulationApp must start before importing any controller / Isaac extension.
import main as entry  # isort: skip

import cv2
import numpy as np
from loguru import logger

from controllers.open_transport_pour_controller import TaskPhase
from experiments.liquid.arm_telemetry import ArmTelemetry
from experiments.liquid.grasp_telemetry import GraspTelemetry

original_create = entry.create_controller
physics_only = os.environ.get("CONICAL_PHYSICS_ONLY") == "1"
telemetry_interval = int(os.environ.get("CONICAL_TELEMETRY_INTERVAL", "60"))
if telemetry_interval < 1:
    raise ValueError("CONICAL_TELEMETRY_INTERVAL must be positive")

if physics_only:
    if not entry.args.no_video:
        raise ValueError("Physics-only diagnostics require --no-video; no training images are generated")
    original_task = entry.create_task
    # Keep the SAME app clock and configured substep batching as production.
    # Disable render products, not the stepping path used by collection/replay.
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport:
        viewport.updates_enabled = False

    def create_physics_task(*args, **kwargs):
        cfg = kwargs["cfg"]
        if cfg.mode != "collect":
            raise ValueError("Physics-only probes do not validate image collection or replay")
        cfg.cameras = []
        cfg.collector.type = "mock"
        return original_task(*args, **kwargs)

    entry.create_task = create_physics_task


def create_controller(*args, **kwargs):
    owner = original_create(*args, **kwargs)
    if owner.mode == "replay":
        owner.replay_run_to_end = True
    out = Path(owner.cfg.multi_run.run_dir)
    if physics_only:
        (out / "physics_only.json").write_text(json.dumps({"physics_only": True, "training_data": False, "video": False}))
    component = os.environ.get("CONICAL_ONLY") == "1"
    if component:
        logger.warning("ISOLATED CONICAL COMPONENT TEST: NOT FULL LEVEL-4 SUCCESS")
        (out / "diagnostic_scope.json").write_text(json.dumps({"component_only": True, "full_level4_score": False}))
        generate = owner._generate_random_sequence

        def sequence():
            generate()
            owner.randomized_sequence = [TaskPhase.PICKING2, TaskPhase.POURING, TaskPhase.TRANSPORTING2]

        if owner.mode == "collect":
            owner._generate_random_sequence = sequence
            sequence()
            owner.current_phase = owner.randomized_sequence[0]
            owner._switch_active_controller()

        def component_success():
            bottle = owner.object_utils.get_geometry_center(object_path="/World/conical_bottle02")
            platform = owner.object_utils.get_geometry_center(object_path="/World/target_plat2")
            size = owner.object_utils.get_object_size(object_path="/World/conical_bottle02")
            support = owner.state["placement_supports"]["/World/target_plat2"]
            return bool(owner.liquid_success() and owner._z_init_conical is not None
                        and owner.state["liquid_transfers"]["pour"]["source_up"][2] >= 0.98
                        and owner._zmax_conical - owner._z_init_conical > 0.05
                        and np.linalg.norm(bottle[:2] - platform[:2]) < 0.06
                        and abs(bottle[2] - size[2] / 2 - support[2]) < 0.012
                        and sum(owner.state["joint_positions"][-2:]) >= 0.074)

        owner._check_success = component_success
    original_step = owner.step
    frames = 0
    grasp_audit = None
    arm_audit = None

    def step(state):
        nonlocal frames, grasp_audit, arm_audit
        if os.environ.get("CONICAL_GRASP_AUDIT") == "1" and grasp_audit is None:
            grasp_audit = GraspTelemetry(owner.robot.prim_path)
        if os.environ.get("CONICAL_ARM_AUDIT") == "1" and arm_audit is None:
            arm_audit = ArmTelemetry(owner.robot)
        if component:
            state["init_state"]["extra"]["diagnostic_component_only"] = True
        result = original_step(state)
        frames += 1
        if frames % telemetry_interval == 0 or result[1]:
            atomic = getattr(owner, "active_controller", None)
            motion = atomic._cspace_controller if atomic is not None else owner.rmp_controller
            joints = motion.articulation_rmp.get_active_joints_subset().get_joint_positions()
            hand, orientation = motion.rmp_flow.get_end_effector_pose(joints)
            metrics = state["liquid_transfers"]["pour"]
            physics_dt = entry.World.instance().get_physics_dt()
            row = {
                "frame": frames, "phase": owner.current_phase.value,
                "atomic_phase": getattr(atomic, "_event", None), "failure": getattr(atomic, "_phase_failure_reason", ""),
                "joints": state["joint_positions"], "hand": hand, "hand_matrix": orientation,
                "gripper_position": state["gripper_position"], "liquid": metrics,
                "drive_targets": owner.robot.get_applied_action().joint_positions,
                "placement_supports": state.get("placement_supports"),
                "gripper_gains": owner.robot._articulation_view.get_gains(),
                "finger_contact_readings": [
                    sensor.get_current_frame() if sensor is not None else None
                    for sensor in owner.robot.get_contact_sensor()
                ],
                "kinematic_attachment": getattr(owner.gripper_control, "grasped_object_path", None),
                "done": result[1], "success": result[2], "component_only": component,
                "mode": owner.mode,
                "replay_actions_consumed": getattr(owner, "_current_action_step", None),
                **({"grasp_audit": grasp_audit.observe(hand, metrics, physics_dt)} if grasp_audit is not None else {}),
                **({"arm_audit": arm_audit.observe(physics_dt)} if arm_audit is not None else {}),
            }
            with (out / "conical_telemetry.jsonl").open("a") as file:
                file.write(json.dumps(row, default=lambda value: np.asarray(value).tolist()) + "\n")
            if (frames % 120 == 0 or result[1]) and state["camera_display"]:
                frames_dir = out / "diagnostic_frames"
                frames_dir.mkdir(exist_ok=True)
                images = [image.transpose(1, 2, 0) for image in state["camera_display"].values()]
                cv2.imwrite(str(frames_dir / f"frame_{frames:05d}.png"), cv2.cvtColor(np.hstack(images), cv2.COLOR_RGB2BGR))
        return result

    owner.step = step
    return owner


entry.create_controller = create_controller
original_video = entry.VideoRecorder
video = None


def create_video(*args, **kwargs):
    global video
    video = original_video(*args, **kwargs)
    return video


entry.VideoRecorder = create_video
try:
    entry.main()
except BaseException:
    # Preserve failure evidence even when an exception or Ctrl-C ends a probe.
    if video is not None:
        video.finish_episode(False)
        video.close()
    entry.simulation_app.close()
    raise
