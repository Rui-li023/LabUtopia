"""Versioned pre-warmup state for fixed-base, position-controlled replay.

Unlike the legacy init_state (poses at reset plus robot joints after warmup),
all values in this snapshot belong to the same simulation instant. Store it
in init_state.extra so legacy datasets retain their original interpretation.
"""

from typing import Any

import numpy as np
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdGeom


def _vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("Replay physics snapshot requires finite vectors")
    return array.tolist()


def _bodies(task: Any) -> dict[str, Any]:
    bodies = {}
    articulations = getattr(task, "LIQUID_REPLAY_ARTICULATIONS", ())
    for offsets in task._object_rigid_body_offsets.values():
        for path, _ in offsets:
            if any(path.startswith(root + "/") for root in articulations):
                continue
            if path not in bodies:
                body = SingleRigidPrim(prim_path=path, reset_xform_properties=False)
                body.initialize()
                bodies[path] = body
    return bodies


def _articulations(task: Any) -> dict[str, Any]:
    result = {}
    for path in getattr(task, "LIQUID_REPLAY_ARTICULATIONS", ()):
        articulation = SingleArticulation(prim_path=path, reset_xform_properties=False)
        articulation.initialize()
        result[path] = articulation
    return result


def _translation_ops(task: Any) -> dict[str, Any]:
    """Read native local translations, including static pivoted supports."""
    result = {}
    for path in task._object_rigid_body_offsets:
        ops = UsdGeom.Xformable(task.stage.GetPrimAtPath(path)).GetOrderedXformOps()
        if not ops or ops[0].GetOpType() != UsdGeom.XformOp.TypeTranslate or ops[0].IsInverseOp():
            raise ValueError(f"Replay snapshot requires a leading translation op: {path}")
        result[path] = ops[0]
    return result


def capture_pre_warmup(task: Any) -> dict[str, Any]:
    """Capture after task reset has placed objects, before any warmup steps."""
    robot = task.robot
    if getattr(robot, "_gripper_control_mode", "position") != "position":
        raise ValueError("Pre-warmup snapshots currently support position grippers only")
    action = robot.get_applied_action()
    if action is None:
        raise ValueError("Cannot capture replay snapshot before robot drives initialize")
    position, orientation = robot.get_world_pose()
    rigid_bodies = {}
    for path, body in _bodies(task).items():
        body_position, body_orientation = body.get_world_pose()
        rigid_bodies[path] = {
            "position": _vector(body_position),
            "orientation_wxyz": _vector(body_orientation),
            "linear_velocity": _vector(body.get_linear_velocity()),
            "angular_velocity": _vector(body.get_angular_velocity()),
        }
    articulations = {}
    for path, articulation in _articulations(task).items():
        pose, quaternion = articulation.get_world_pose()
        articulations[path] = {
            "position": _vector(pose), "orientation_wxyz": _vector(quaternion),
            "joint_positions": _vector(articulation.get_joint_positions()),
            "joint_velocities": _vector(articulation.get_joint_velocities()),
        }
    return {
        "version": 1,
        "timing": "pre_warmup",
        "gripper_command": "absolute_position_v1",
        # Legacy object_pose_positions is float32 WORLD space. Its precision
        # and pivot correction are insufficient for exact contact replay. Keep
        # authored LOCAL translations of static supports as well as rigid bodies.
        "root_translations": {path: _vector(op.Get()) for path, op in _translation_ops(task).items()},
        "robot": {
            "position": _vector(position),
            "orientation_wxyz": _vector(orientation),
            "joint_positions": _vector(robot.get_joint_positions()),
            "joint_velocities": _vector(robot.get_joint_velocities()),
            "drive_positions": _vector(action.joint_positions),
            "drive_velocities": _vector(action.joint_velocities),
            "drive_efforts": _vector(action.joint_efforts),
        },
        "rigid_bodies": rigid_bodies,
        **({"articulations": articulations} if articulations else {}),
    }


def restore_pre_warmup(task: Any, snapshot: dict[str, Any]) -> None:
    """Restore the snapshot once at reset; normal warmup then runs unchanged."""
    if snapshot.get("version") != 1 or snapshot.get("timing") != "pre_warmup":
        raise ValueError("Unsupported replay physics snapshot version or timing")
    if getattr(task.robot, "_gripper_control_mode", "position") != "position":
        raise ValueError("Pre-warmup snapshots currently support position grippers only")
    bodies = _bodies(task)
    if set(bodies) != set(snapshot["rigid_bodies"]):
        raise ValueError("Replay snapshot rigid bodies do not match the task")
    articulations = _articulations(task)
    if set(articulations) != set(snapshot.get("articulations", {})):
        raise ValueError("Replay snapshot scene articulations do not match the task")
    translations = snapshot.get("root_translations")
    if translations is not None:
        ops = _translation_ops(task)
        if set(ops) != set(translations):
            raise ValueError("Replay snapshot object roots do not match the task")
        for path, op in ops.items():
            # Preserve the authored op type/precision; do not replace xform
            # stacks or reinterpret world coordinates as local translations.
            op.Set(type(op.Get())(*translations[path]))
    robot = task.robot
    state = snapshot["robot"]
    robot.set_world_pose(position=np.asarray(state["position"]), orientation=np.asarray(state["orientation_wxyz"]))
    robot.set_joint_positions(np.asarray(state["joint_positions"]))
    robot.set_joint_velocities(np.asarray(state["joint_velocities"]))
    robot.apply_action(
        ArticulationAction(
            joint_positions=None if state["drive_positions"] is None else np.asarray(state["drive_positions"]),
            joint_velocities=None if state["drive_velocities"] is None else np.asarray(state["drive_velocities"]),
            joint_efforts=None if state["drive_efforts"] is None else np.asarray(state["drive_efforts"]),
        )
    )
    for path, body in bodies.items():
        state = snapshot["rigid_bodies"][path]
        body.set_world_pose(position=np.asarray(state["position"]), orientation=np.asarray(state["orientation_wxyz"]))
        body.set_linear_velocity(np.asarray(state["linear_velocity"]))
        body.set_angular_velocity(np.asarray(state["angular_velocity"]))
    for path, articulation in articulations.items():
        state = snapshot["articulations"][path]
        articulation.set_world_pose(
            position=np.asarray(state["position"]), orientation=np.asarray(state["orientation_wxyz"])
        )
        articulation.set_joint_positions(np.asarray(state["joint_positions"]))
        articulation.set_joint_velocities(np.asarray(state["joint_velocities"]))
