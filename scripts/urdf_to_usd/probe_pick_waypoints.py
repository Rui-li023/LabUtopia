"""Measure RMPFlow tracking over a pick path without task-scene contacts.

This is a bring-up diagnostic, not a replacement for the physical pick test.  It
uses the same grasp-frame convention and waypoint geometry as PickController so
candidate wrist orientations can be rejected before spending time on collision
and gripper tuning.

Example (Isaac Sim Python)::

    python scripts/urdf_to_usd/probe_pick_waypoints.py \
        --robot xarm6_robotiq --base=-0.35,0,0.71 \
        --object=0.235,-0.06,0.764 --object-height 0.111 \
        --euler 0,90,-25 --euler 0,105,-25 --euler 0,120,-25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


def _vector(value: str, *, length: int = 3) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split(",")], dtype=float)
    if values.shape != (length,):
        raise argparse.ArgumentTypeError(f"expected {length} comma-separated values, got {value!r}")
    return values


def _arm_vector(value: str) -> np.ndarray:
    return _vector(value, length=6)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        default="xarm6_robotiq",
        help="Robot factory registry key",
    )
    parser.add_argument("--base", type=_vector, default=_vector("-0.35,0,0.71"))
    parser.add_argument(
        "--approach",
        type=_vector,
        default=None,
        help="Explicit world-frame pre-grasp direction; defaults to object-to-base",
    )
    parser.add_argument(
        "--bearing-deg",
        type=float,
        default=None,
        help="Explicit grasp-frame world yaw; defaults to the object bearing from --base",
    )
    parser.add_argument("--object", type=_vector, default=_vector("0.235,-0.06,0.764"))
    parser.add_argument("--object-height", type=float, default=0.111)
    parser.add_argument("--pick-z-offset", type=float, default=0.045)
    parser.add_argument("--pre-offset-x", type=float, default=0.05)
    parser.add_argument("--pre-offset-z", type=float, default=0.12)
    parser.add_argument("--pre-lower-z", type=float, default=0.05)
    parser.add_argument("--lift-z", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument(
        "--initial-arm",
        type=_arm_vector,
        default=None,
        help="Optional six-joint arm pose used for every candidate reset",
    )
    parser.add_argument(
        "--euler",
        action="append",
        type=_vector,
        help="Canonical scipy xyz Euler degrees; repeat to scan candidates",
    )
    return parser.parse_args()


ARGS = _parse_args()
APP = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from factories.robot_factory import create_robot  # noqa: E402
from robots.base_robot import BaseRobot  # noqa: E402
from robots.franka.rmpflow_controller import RMPFlowController  # noqa: E402


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _tcp_position(robot: BaseRobot) -> np.ndarray:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(robot.gripper_center_prim_path)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = transform.ExtractTranslation()
    return np.asarray(translation, dtype=float)


def _orientation_wxyz(
    euler_deg: np.ndarray,
    bearing_deg: float,
    tool_correction_euler_deg: np.ndarray,
) -> np.ndarray:
    nominal = R.from_euler("xyz", euler_deg, degrees=True)
    correction = R.from_euler("xyz", tool_correction_euler_deg, degrees=True)
    orientation = R.from_euler("z", bearing_deg, degrees=True) * nominal * correction
    xyzw = orientation.as_quat()
    return xyzw[[3, 0, 1, 2]]


def _waypoints() -> list[tuple[str, np.ndarray]]:
    obj = ARGS.object.copy()
    if ARGS.approach is None:
        horizontal = obj[:2] - ARGS.base[:2]
        approach = np.zeros(3)
        approach[:2] = -horizontal / np.linalg.norm(horizontal)
    else:
        approach = ARGS.approach.copy()
        norm = np.linalg.norm(approach)
        if norm <= 1e-8:
            raise ValueError("--approach must be non-zero")
        approach /= norm

    pre_high = obj + approach * ARGS.pre_offset_x
    pre_high[2] += ARGS.object_height + ARGS.pre_offset_z
    pre_low = obj + approach * ARGS.pre_offset_x
    pre_low[2] += ARGS.pre_lower_z
    grasp = obj.copy()
    grasp[2] += ARGS.pick_z_offset
    lift = grasp.copy()
    lift[2] += ARGS.lift_z
    return [
        ("pre_high", pre_high),
        ("pre_low", pre_low),
        ("grasp", grasp),
        ("lift", lift),
    ]


def _reset_robot(robot: BaseRobot, controller: RMPFlowController) -> None:
    positions = robot.DEFAULT_JOINT_POSITIONS.copy()
    if ARGS.initial_arm is not None:
        arm_indices = np.asarray(robot.get_arm_joint_indices(), dtype=int)
        if ARGS.initial_arm.shape != (arm_indices.size,):
            raise ValueError(
                f"--initial-arm has {ARGS.initial_arm.size} values, but {robot.name} has {arm_indices.size} arm joints"
            )
        positions[arm_indices] = ARGS.initial_arm
    robot.set_joint_positions(positions)
    robot.set_joint_velocities(np.zeros_like(positions))
    robot.apply_action(ArticulationAction(joint_positions=positions))
    controller.reset()
    for step in range(30):
        WORLD.step(render=step == 29)


def _commanded_arm_positions(action, arm_indices: np.ndarray) -> np.ndarray | None:
    """Return action targets in arm-joint order for branched articulations."""
    if action.joint_positions is None:
        return None
    commanded = np.asarray(action.joint_positions, dtype=float).reshape(-1)
    if action.joint_indices is None:
        if commanded.size == ROBOT.num_dof:
            return commanded[arm_indices]
        if commanded.size == arm_indices.size:
            return commanded
        return None

    action_indices = np.asarray(action.joint_indices, dtype=int).reshape(-1)
    by_index = dict(zip(action_indices.tolist(), commanded.tolist(), strict=True))
    if not all(int(index) in by_index for index in arm_indices):
        return None
    return np.asarray([by_index[int(index)] for index in arm_indices], dtype=float)


def _run_candidate(
    robot: BaseRobot,
    controller: RMPFlowController,
    euler_deg: np.ndarray,
    bearing_deg: float,
) -> tuple[float, float]:
    _reset_robot(robot, controller)
    orientation = _orientation_wxyz(
        euler_deg,
        bearing_deg,
        np.asarray(robot.tool_frame_correction_euler_deg, dtype=float),
    )
    worst_tcp_error = 0.0
    worst_joint_error = 0.0
    arm_indices = np.asarray(robot.get_arm_joint_indices(), dtype=int)

    _say(f"[candidate] euler_deg={euler_deg.tolist()} bearing_deg={bearing_deg:.3f}")
    for label, target in _waypoints():
        waypoint_joint_error = 0.0
        for step in range(ARGS.steps):
            action = controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orientation,
            )
            measured = np.asarray(robot.get_joint_positions(), dtype=float)
            commanded_arm = _commanded_arm_positions(action, arm_indices)
            if commanded_arm is not None:
                error = np.abs(commanded_arm - measured[arm_indices])
                waypoint_joint_error = max(waypoint_joint_error, float(np.max(error)))
            robot.apply_action(action)
            # Dynamic-link xforms are synchronized back to USD on a render step.
            # The real task renders every frame; one final synchronized frame is
            # enough here to make the measured tool_frame authoritative without
            # turning a large orientation scan into a renderer benchmark.
            WORLD.step(render=step == ARGS.steps - 1)

        tcp = _tcp_position(robot)
        tcp_error = float(np.linalg.norm(tcp - target))
        measured = np.asarray(robot.get_joint_positions(), dtype=float)[arm_indices]
        worst_tcp_error = max(worst_tcp_error, tcp_error)
        worst_joint_error = max(worst_joint_error, waypoint_joint_error)
        _say(
            f"  [{label:8s}] target={np.round(target, 4).tolist()} "
            f"tcp={np.round(tcp, 4).tolist()} tcp_err={tcp_error:.5f} "
            f"peak_joint_err={waypoint_joint_error:.5f} q={np.round(measured, 3).tolist()}"
        )

    _say(
        f"[summary] euler_deg={euler_deg.tolist()} worst_tcp_err={worst_tcp_error:.5f} "
        f"peak_joint_err={worst_joint_error:.5f}"
    )
    return worst_tcp_error, worst_joint_error


if ARGS.steps <= 0:
    raise ValueError("--steps must be positive")

EULERS = ARGS.euler or [
    _vector("0,90,-25"),
    _vector("0,90,0"),
    _vector("0,90,25"),
    _vector("0,105,-25"),
    _vector("0,105,0"),
    _vector("0,105,25"),
    _vector("0,120,-25"),
    _vector("0,120,0"),
    _vector("0,120,25"),
]

WORLD = World(stage_units_in_meters=1.0)
WORLD.scene.add_default_ground_plane()
ROBOT = create_robot(
    ARGS.robot,
    prim_path="/World/ProbeRobot",
    name=f"{ARGS.robot}_probe",
    position=ARGS.base,
)
WORLD.scene.add(ROBOT)
WORLD.reset()
ROBOT.initialize()

# Match ``main.py`` when the pick config has ``collect_position_only: true``.
RMPFlowController.ignore_robot_state_updates = True
CONTROLLER = RMPFlowController(
    name=f"{ARGS.robot}_rmpflow_probe",
    robot_articulation=ROBOT,
    physics_dt=1.0 / 60.0,
)

DELTA = ARGS.object[:2] - ARGS.base[:2]
BEARING_DEG = (
    float(ARGS.bearing_deg) if ARGS.bearing_deg is not None else float(np.degrees(np.arctan2(DELTA[1], DELTA[0])))
)
_say(
    f"[setup] base={ARGS.base.tolist()} object={ARGS.object.tolist()} "
    f"bearing_deg={BEARING_DEG:.3f} steps_per_waypoint={ARGS.steps}"
)

RESULTS = []
for euler in EULERS:
    RESULTS.append((*_run_candidate(ROBOT, CONTROLLER, euler, BEARING_DEG), euler))

_say("[ranking]")
for tcp_error, joint_error, euler in sorted(RESULTS, key=lambda item: (item[0], item[1])):
    _say(f"  euler_deg={euler.tolist()} worst_tcp_err={tcp_error:.5f} peak_joint_err={joint_error:.5f}")

WORLD.stop()
APP.close()
