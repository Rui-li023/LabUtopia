"""Shared pipette path planning, with actual geometry and unchanged collisions.

Import after SimulationApp startup. Generated sphere descriptors are planning
artifacts only; this module never modifies robot/scene asset collision shapes.
The fixed-scene probe imports these shared primitives through a compatibility shim.
"""

from itertools import pairwise, product
from pathlib import Path

import numpy as np
import yaml
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import interface_config_loader
from isaacsim.robot_motion.motion_generation.lula.path_planners import RRT
from scipy.spatial.transform import Rotation

from controllers.pipette_clearance import (
    PlanningBox,
    RackArmClearance,
    cover_box_with_spheres,
    cover_convex_mesh_with_spheres,
    oriented_mesh_bounds,
)
from controllers.pipette_kinematics import continuous_inverse_kinematics


def world_mesh_vertices(usd, stage, root_path):
    """Read every mesh below the real prim, including visual protrusions."""
    cache = usd.UsdGeom.XformCache()
    groups = []
    for prim in usd.Usd.PrimRange(stage.GetPrimAtPath(root_path), usd.Usd.TraverseInstanceProxies()):
        if prim.IsA(usd.UsdGeom.Mesh):
            matrix = cache.GetLocalToWorldTransform(prim)
            groups.extend(matrix.Transform(usd.Gf.Vec3d(*p)) for p in usd.UsdGeom.Mesh(prim).GetPointsAttr().Get())
    if not groups:
        raise ValueError(f"No actual mesh vertices found for carried geometry: {root_path}")
    return np.asarray(groups, dtype=float)


def world_collision_meshes(usd, stage, root_path):
    """Read each enabled physical mesh separately; exclude visual-only geometry."""
    cache = usd.UsdGeom.XformCache()
    groups = {}
    for prim in usd.Usd.PrimRange(stage.GetPrimAtPath(root_path), usd.Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(usd.UsdPhysics.CollisionAPI):
            continue
        if not usd.UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        if not prim.IsA(usd.UsdGeom.Mesh):
            raise ValueError(f"Unsupported physical hand collider shape: {prim.GetPath()}")
        matrix = cache.GetLocalToWorldTransform(prim)
        groups[str(prim.GetPath())] = np.asarray(
            [matrix.Transform(usd.Gf.Vec3d(*p)) for p in usd.UsdGeom.Mesh(prim).GetPointsAttr().Get()], dtype=float
        )
    if not groups:
        raise ValueError(f"No enabled physical mesh colliders: {root_path}")
    return groups


def screen_joint_path(path, checker, limits, *, joint_margin):
    """Check every <=0.01 rad segment sample, not just RRT vertices."""
    path, limits = np.asarray(path, dtype=float), np.asarray(limits, dtype=float)
    if (
        path.ndim != 2
        or path.shape[1] != 7
        or len(path) == 0
        or limits.shape != (7, 2)
        or not np.isfinite(path).all()
        or not np.isfinite(limits).all()
        or np.any(limits[:, 1] <= limits[:, 0])
        or not np.isfinite(joint_margin)
        or joint_margin < 0
    ):
        raise ValueError("Expected a finite seven-joint path and physical bounds")
    dense = [path[0]]
    for first, last in pairwise(path):
        count = max(1, int(np.ceil(np.max(np.abs(last - first)) / 0.01)))
        dense.extend(first + t * (last - first) for t in np.linspace(0, 1, count + 1)[1:])
    clearances = [checker.minimum_clearance(q) for q in dense]
    limiting = int(np.argmin(clearances))
    clearance = clearances[limiting]
    margin = float(np.min(np.minimum(path - limits[:, 0], limits[:, 1] - path)))
    return {
        "accepted": bool(np.isfinite(clearance) and clearance >= checker.margin and margin >= joint_margin),
        "minimum_gap": float(clearance),
        "minimum_joint_margin": margin,
        "dense_samples": len(dense),
        "start_gap": float(clearances[0]),
        "end_gap": float(clearances[-1]),
        "limiting_joints": dense[limiting].tolist(),
        "limiting_sample": limiting,
    }


def shortcut_joint_path(path, checker, limits, *, joint_margin):
    """Remove redundant corners only if each replacement chord passes screening."""
    path = np.asarray(path, dtype=float)
    if not screen_joint_path(path, checker, limits, joint_margin=joint_margin)["accepted"]:
        raise ValueError("Cannot shortcut an invalid input path")
    result = [path[0].copy()]
    index, probes = 0, 0
    while index < len(path) - 1:
        selected = index + 1
        for candidate in range(len(path) - 1, index + 1, -1):
            if probes >= 64:
                break
            probes += 1
            if screen_joint_path(path[[index, candidate]], checker, limits, joint_margin=joint_margin)["accepted"]:
                selected = candidate
                break
        result.append(path[selected].copy())
        index = selected
    return np.asarray(result)


def screened_hold_compensation(target, measured, bias, checker, limits, *, joint_margin):
    """Bounded position-only compensation while holding the SAME endpoint.

    The original Cartesian arrival target is unchanged. Never apply a bias
    through a large tracking error, a joint limit or a blocked measured-to-
    command segment. This is not a motor-effort or gravity-model estimate.
    """
    target, measured, bias = (np.asarray(value, dtype=float) for value in (target, measured, bias))
    limits = np.asarray(limits, dtype=float)
    if any(value.shape != (7,) or not np.isfinite(value).all() for value in (target, measured, bias)):
        raise ValueError("Hold compensation needs finite seven-joint vectors")
    if np.any(np.abs(bias) > 0.03 + 1e-12) or np.max(np.abs(target - measured)) >= 0.06:
        raise ValueError("Hold compensation exceeds its bounded tracking envelope")
    if limits.shape != (7, 2) or not np.isfinite(limits).all() or np.any(limits[:, 1] <= limits[:, 0]):
        raise ValueError("Hold compensation needs finite physical joint bounds")
    if not np.isfinite(joint_margin) or joint_margin < 0 or np.any(limits[:, 1] - limits[:, 0] <= 2 * joint_margin):
        raise ValueError("Hold compensation needs a valid physical joint margin")
    if np.any(target < limits[:, 0] + joint_margin) or np.any(target > limits[:, 1] - joint_margin):
        raise ValueError("Original hold endpoint violates the physical joint margin")
    next_bias = np.clip(bias + np.clip(0.08 * (target - measured), -0.0008, 0.0008), -0.03, 0.03)
    # Stay just inside the margin so subtraction roundoff cannot turn a
    # nominal boundary command into an out-of-bounds screening result.
    command = np.clip(
        target + next_bias,
        np.nextafter(limits[:, 0] + joint_margin, limits[:, 1]),
        np.nextafter(limits[:, 1] - joint_margin, limits[:, 0]),
    )
    if np.any(np.abs(command - target) > 0.03 + 1e-12):
        raise ValueError("Original endpoint lies outside the bounded physical command envelope")
    verdict = screen_joint_path(np.asarray([measured, command]), checker, limits, joint_margin=joint_margin)
    if not verdict["accepted"]:
        raise ValueError("Hold compensation would violate the original dense clearance/joint screen")
    return command, command - target, verdict


def prepare_approach_path(path, checker, limits, interpolate, *, joint_margin):
    """Shortcut with full geometry, then validate actual execution samples."""
    shortened = shortcut_joint_path(path, checker, limits, joint_margin=joint_margin)
    commands = interpolate(shortened)
    verdict = screen_joint_path(commands, checker, limits, joint_margin=joint_margin)
    if not verdict["accepted"]:
        raise ValueError("Shortened approach interpolation violates physical clearance or joint margin")
    return shortened, {
        "original_path": np.asarray(path).tolist(),
        "original_waypoints": len(path),
        "shortened_waypoints": len(shortened),
        "execution_commands": len(commands),
        "minimum_command_gap": verdict["minimum_gap"],
        "minimum_command_joint_margin": verdict["minimum_joint_margin"],
    }


def rank_pose_ik(solver, checker, frame, position, orientation, start, limits, seed, *, joint_margin):
    """Bounded alternative IK endpoints; endpoint feasibility is NOT a path."""
    limits, start = np.asarray(limits), np.asarray(start)
    target_rotation = Rotation.from_quat(np.asarray(orientation)[[1, 2, 3, 0]])
    seeds = [start] + [
        np.array([shoulder, -0.7, -shoulder, -2.2, wrist, 2.4, roll])
        for shoulder, wrist, roll in product((-1.5, 0.0, 1.5), (-2.0, 0.0, 2.0), (-2.0, 0.0, 2.0))
    ]
    seeds.extend(np.random.default_rng(seed).uniform(limits[:, 0] + joint_margin, limits[:, 1] - joint_margin, (64, 7)))
    candidates = []
    for warm_start in seeds:
        joints, ok = solver.compute_inverse_kinematics(
            frame, position, orientation, warm_start=warm_start, position_tolerance=0.0005, orientation_tolerance=0.005
        )
        joints = np.asarray(joints)
        if not ok or joints.shape != (7,) or not np.isfinite(joints).all():
            continue
        margin = float(np.min(np.minimum(joints - limits[:, 0], limits[:, 1] - joints)))
        if margin < joint_margin or any(np.linalg.norm(joints - item["joints"]) < 0.02 for item in candidates):
            continue
        actual_position, actual_matrix = solver.compute_forward_kinematics(frame, joints)
        if not np.isfinite(actual_position).all() or not np.isfinite(actual_matrix).all():
            continue
        angle = (Rotation.from_matrix(actual_matrix) * target_rotation.inv()).magnitude()
        if np.linalg.norm(np.asarray(actual_position).reshape(3) - position) > 0.0005 or angle > 0.005:
            continue
        gap = checker.minimum_clearance(joints)
        if np.isfinite(gap) and gap >= checker.margin:
            candidates.append(
                {"joints": joints.copy(), "joint_cost": float(np.max(np.abs(joints - start))), "endpoint_gap": gap}
            )
    return sorted(candidates, key=lambda item: item["joint_cost"])


def screen_pose_continuation(
    solver, solve_local, frame, start, targets, orientation, limits, *, joint_margin, geometry_checker=None
):
    """Forecast continuous insertion/retreat IK and optional robot-only clearance.

    Fixed orientation, <=2 mm Cartesian samples, <=256 local solves, original
    physical joint margin and pose tolerances. No robot or object is actuated.
    Intended seated tool/rack contact requires a separate physical evaluation.
    The optional checker covers the robot with OPEN fingers against the rack,
    bench and other tools, excluding only the currently held tool. It never
    disables physical contacts or interprets a forecast as placement success.
    """
    start, targets, limits = (np.asarray(value, dtype=float) for value in (start, targets, limits))
    orientation = np.asarray(orientation, dtype=float)
    if (
        start.shape != (7,)
        or targets.ndim != 2
        or targets.shape[1] != 3
        or not 1 <= len(targets) <= 4
        or limits.shape != (7, 2)
        or orientation.shape != (4,)
        or not all(np.isfinite(value).all() for value in (start, targets, limits, orientation))
        or not np.isclose(np.linalg.norm(orientation), 1, atol=1e-6)
        or not np.isfinite(joint_margin)
        or not 0.05 <= joint_margin <= 0.25
        or np.any(limits[:, 1] - limits[:, 0] <= 2 * joint_margin)
    ):
        raise ValueError("Continuation screen requires finite bounded joints and Cartesian targets")
    margin = float(np.min(np.minimum(start - limits[:, 0], limits[:, 1] - start)))
    record = {
        "accepted": False,
        "samples": 0,
        "minimum_joint_margin": margin,
        "kinematics_only": geometry_checker is None,
    }
    if geometry_checker is not None:
        initial = screen_joint_path([start], geometry_checker, limits, joint_margin=joint_margin)
        record["minimum_robot_gap_m"] = initial["minimum_gap"]
        if not initial["accepted"]:
            return {**record, "reason": "Starting endpoint fails open-hand placement geometry", "geometry": initial}
    if margin < joint_margin:
        return {**record, "reason": "Starting endpoint violates physical joint margin"}
    position, _ = solver.compute_forward_kinematics(frame, start)
    position = np.asarray(position).reshape(3)
    if not np.isfinite(position).all():
        return {**record, "reason": "Starting endpoint FK is nonfinite"}
    segments = list(pairwise(np.vstack([position, targets])))
    counts = [max(1, int(np.ceil(np.linalg.norm(last - first) / 0.002))) for first, last in segments]
    if sum(counts) > 256:
        return {**record, "reason": "Continuation exceeds bounded 256-sample forecast"}
    rotation = Rotation.from_quat(orientation[[1, 2, 3, 0]])
    joints = start.copy()
    for segment, ((first, last), count) in enumerate(zip(segments, counts)):
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            target = first + fraction * (last - first)
            next_joints, ok = solve_local(solver, frame, target, orientation, joints, limits, joint_margin=joint_margin)
            next_joints = np.asarray(next_joints)
            record.update(samples=record["samples"] + 1, segment=segment, target=target.tolist())
            if not ok or next_joints.shape != (7,) or not np.isfinite(next_joints).all():
                return {**record, "reason": "No local continuation IK solution"}
            margin = float(np.min(np.minimum(next_joints - limits[:, 0], limits[:, 1] - next_joints)))
            record["minimum_joint_margin"] = min(record["minimum_joint_margin"], margin)
            if margin < joint_margin or np.max(np.abs(next_joints - joints)) > 0.35:
                return {**record, "reason": "Continuation violates joint margin or local branch trust region"}
            actual, matrix = solver.compute_forward_kinematics(frame, next_joints)
            if not np.isfinite(actual).all() or not np.isfinite(matrix).all():
                return {**record, "reason": "Continuation FK is nonfinite"}
            if (
                np.linalg.norm(np.asarray(actual).reshape(3) - target) > 0.0005
                or (Rotation.from_matrix(matrix) * rotation.inv()).magnitude() > 0.005
            ):
                return {**record, "reason": "Continuation FK violates original pose tolerances"}
            if geometry_checker is not None:
                geometry = screen_joint_path([joints, next_joints], geometry_checker, limits, joint_margin=joint_margin)
                record["minimum_robot_gap_m"] = min(record["minimum_robot_gap_m"], geometry["minimum_gap"])
                if not geometry["accepted"]:
                    return {
                        **record,
                        "reason": "Insertion/retreat intersects open-hand placement geometry",
                        "geometry": geometry,
                    }
            joints = next_joints
    return {**record, "accepted": True, "final_joints": joints.tolist()}


def improve_pose_path(
    planner,
    checker,
    frame,
    position,
    orientation,
    start,
    baseline,
    limits,
    interpolate,
    seed,
    *,
    joint_margin,
    endpoint_validator=None,
):
    """Compare at most eight native RRT joint-target plans for the SAME pose.

    Retain a valid baseline unless a fully screened candidate has fewer actual
    interpolation commands. Native planning is still required even when a
    straight chord appears externally clear. This is not an execution result.
    """
    best, best_count = baseline, None if baseline is None else len(interpolate(baseline))
    record = {
        "baseline_path": None if baseline is None else np.asarray(baseline).tolist(),
        "baseline_commands": best_count,
        "selected_commands": best_count,
        "improved": False,
        "candidates": [],
    }
    if (
        baseline is not None
        and not screen_joint_path(interpolate(baseline), checker, limits, joint_margin=joint_margin)["accepted"]
    ):
        raise ValueError("Cannot optimize an invalid baseline path")
    if baseline is not None and endpoint_validator is not None:
        record["baseline_continuation"] = endpoint_validator(np.asarray(baseline)[-1])
        if not record["baseline_continuation"]["accepted"]:
            best, best_count = None, None
            record["selected_commands"] = None
    candidates = rank_pose_ik(
        checker.solver, checker, frame, position, orientation, start, limits, seed, joint_margin=joint_margin
    )
    record["feasible_unique_endpoints"] = len(candidates)
    target_rotation = Rotation.from_quat(np.asarray(orientation)[[1, 2, 3, 0]])
    try:
        planned = 0
        for index, candidate in enumerate(candidates[: 32 if endpoint_validator is not None else 8]):
            detail = {**candidate, "joints": candidate["joints"].tolist(), "accepted": False, "path_found": False}
            record["candidates"].append(detail)
            lower_bound = len(interpolate(np.asarray([start, candidate["joints"]])))
            if best_count is not None and lower_bound >= best_count:
                detail["reason"] = "Direct interpolation lower bound cannot improve current path"
                continue
            if endpoint_validator is not None:
                detail["continuation"] = endpoint_validator(candidate["joints"])
                if not detail["continuation"]["accepted"]:
                    detail["reason"] = "Endpoint cannot continuously complete insertion and retreat"
                    continue
            if planned >= 8:
                detail["reason"] = "Bounded native-planning budget exhausted"
                break
            planned += 1
            planner.set_cspace_target(candidate["joints"])
            planner.set_random_seed(seed + index)
            path = planner.compute_path(np.asarray(start), np.empty(0))
            detail["path_found"] = path is not None
            if path is None:
                continue
            path = np.asarray(path)
            if (
                path.ndim != 2
                or path.shape[1] != 7
                or len(path) < 2
                or not np.isfinite(path).all()
                or not np.allclose(path[0], start, atol=1e-6, rtol=0)
            ):
                detail["reason"] = "Native path does not start at the measured robot configuration"
                continue
            actual_position, actual_matrix = checker.solver.compute_forward_kinematics(frame, path[-1])
            if (
                not np.isfinite(actual_position).all()
                or not np.isfinite(actual_matrix).all()
                or np.linalg.norm(np.asarray(actual_position).reshape(3) - position) > 0.0005
                or (Rotation.from_matrix(actual_matrix) * target_rotation.inv()).magnitude() > 0.005
            ):
                detail["reason"] = "Native path endpoint violates original pose tolerance"
                continue
            if endpoint_validator is not None and not np.allclose(path[-1], candidate["joints"], atol=1e-6, rtol=0):
                detail["actual_endpoint_continuation"] = endpoint_validator(path[-1])
                if not detail["actual_endpoint_continuation"]["accepted"]:
                    detail["reason"] = "Actual native path endpoint fails continuation"
                    continue
            try:
                shortened, screening = prepare_approach_path(
                    path, checker, limits, interpolate, joint_margin=joint_margin
                )
            except ValueError as error:
                detail["reason"] = str(error)
                continue
            detail.update(screening)
            count = screening["execution_commands"]
            if best_count is None or count < best_count:
                best, best_count = shortened, count
                detail["accepted"] = True
                record.update(improved=True, selected_commands=count, selected_path=shortened.tolist())
    finally:
        planner.set_end_effector_target(position, orientation)
    return best, record


def forecast_seated_pose(active, state, pick_orientation, tcp_local):
    """Forecast a seat target before closure; measured post-lift IK still required."""
    correction = Rotation.from_quat(state["target_quaternion"]) * Rotation.from_quat(state["object_quaternion"]).inv()
    pick_tcp = np.asarray(state["object_grasp_position"]) + np.array(
        [0, 0, float(active.cfg.task.pipette_grasp_height_offset)]
    )
    hand = correction * Rotation.from_quat(np.asarray(pick_orientation)[[1, 2, 3, 0]])
    tcp = (
        active._command_seat_position(state)
        + correction.apply(pick_tcp - state["object_reference_position"])
        + [0, 0, active._release_clearance()]
    )
    return tcp - hand.apply(tcp_local), hand.as_quat()[[3, 0, 1, 2]]


def grasp_orientation_candidates(preferred, axis, rolls, preferred_roll, *, symmetric=False):
    """Swap opposing jaws about LOCAL approach axis, not arbitrary world vertical."""
    axis = np.array(axis, dtype=float, copy=True)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError("Expected a finite nonzero pipette axis")
    axis /= np.linalg.norm(axis)
    preferred = Rotation.from_quat(np.asarray(preferred)[[1, 2, 3, 0]])
    for roll in rolls:
        hand = Rotation.from_rotvec(axis * np.radians(roll - preferred_roll)) * preferred
        for swap in (False, True) if symmetric else (False,):
            candidate = hand * Rotation.from_euler("z", 180 if swap else 0, degrees=True)
            yield roll, swap, candidate.as_quat()[[3, 0, 1, 2]]


def seat_heading_command_state(state, heading_deg):
    """Separate motion target, NEVER a modified observation or success target.

    A bounded rotation about the rack's own upright axis preserves that axis
    and the selected seat position. Native physics must still validate fit;
    all existing release tolerances and original task-success checks remain.
    """
    if not np.isfinite(heading_deg) or not -20 <= heading_deg <= 20:
        raise ValueError("Commanded seat heading must be finite and within +/-20 degrees")
    if heading_deg == 0:
        return state
    quaternion = np.asarray(state["target_quaternion"], dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion), 1):
        raise ValueError("Commanded seat heading requires a normalized observed rack quaternion")
    commanded = Rotation.from_quat(quaternion) * Rotation.from_euler("z", heading_deg, degrees=True)
    return {**state, "target_quaternion": commanded.as_quat()}


def choose_seat_grasp(
    active,
    state,
    preferred,
    tcp_local,
    rolls,
    preferred_roll,
    seed,
    emit,
    *,
    symmetric=False,
    pregrasp_checker=None,
    pregrasp_height=0.0,
    seat_headings=(0.0,),
    on_heading_selected=None,
):
    """Prefer a grip roll, then a feasible bounded heading; neither proves a path."""
    if not 1 <= len(seat_headings) <= 9 or any(not np.isfinite(yaw) or not -20 <= yaw <= 20 for yaw in seat_headings):
        raise ValueError("Expected 1..9 finite seat headings within +/-20 degrees")
    solver = active.rmp_controller.rmp_flow.get_kinematics_solver()
    frame = active.rmp_controller.rmp_flow.end_effector_frame_name
    bounds = np.asarray(active.robot._articulation_view.get_dof_limits())[0, :7]
    margin = float(active.cfg.task.pipette_preflight_joint_margin)
    current = active.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
    seeds = [current] + [
        np.array([shoulder, -0.7, -shoulder, -2.2, wrist, 2.4, roll])
        for shoulder, wrist, roll in product((-1.5, 0.0, 1.5), (-2.0, 0.0, 2.0), (-2.0, 0.0, 2.0))
    ]
    seeds.extend(np.random.default_rng(seed).uniform(bounds[:, 0] + margin, bounds[:, 1] - margin, size=(64, 7)))
    for roll, swapped, pick in grasp_orientation_candidates(
        preferred, state["object_axis"], rolls, preferred_roll, symmetric=symmetric
    ):
        hover_feasible, hover_gap = None, None
        for heading in seat_headings:
            command_state = seat_heading_command_state(state, heading)
            position, orientation = forecast_seated_pose(active, command_state, pick, tcp_local)
            accepted = False
            for start in seeds:
                solution, ok = solver.compute_inverse_kinematics(
                    frame, position, orientation, warm_start=start, position_tolerance=0.001, orientation_tolerance=0.01
                )
                gap = float(np.min(np.minimum(solution - bounds[:, 0], bounds[:, 1] - solution)))
                if ok and gap >= margin:
                    accepted = True
                    break
            seat_feasible = accepted
            if accepted and pregrasp_checker is not None:
                if hover_feasible is None:
                    hover_feasible = False
                    hover_target = (
                        np.asarray(state["object_grasp_position"])
                        + np.array([0, 0, float(active.cfg.task.pipette_grasp_height_offset) + pregrasp_height])
                        - Rotation.from_quat(pick[[1, 2, 3, 0]]).apply(tcp_local)
                    )
                    for start in seeds:
                        hover_joints, hover_ok = solver.compute_inverse_kinematics(
                            frame,
                            hover_target,
                            pick,
                            warm_start=start,
                            position_tolerance=0.0005,
                            orientation_tolerance=0.005,
                        )
                        if (
                            not hover_ok
                            or np.any(hover_joints < bounds[:, 0] + margin)
                            or np.any(hover_joints > bounds[:, 1] - margin)
                        ):
                            continue
                        candidate_gap = pregrasp_checker.minimum_clearance(hover_joints)
                        hover_gap = candidate_gap if hover_gap is None else max(hover_gap, candidate_gap)
                        if candidate_gap >= pregrasp_checker.margin:
                            hover_feasible = True
                            break
                accepted = hover_feasible
            emit(
                {
                    "type": "grasp_seat_forecast",
                    "object_path": state["object_path"],
                    "roll_deg": roll,
                    "jaws_swapped": swapped,
                    "seat_heading_deg": heading,
                    "observed_target_quaternion": np.asarray(state["target_quaternion"]).tolist(),
                    "command_target_quaternion": np.asarray(command_state["target_quaternion"]).tolist(),
                    "feasible": accepted,
                    "seat_feasible": seat_feasible,
                    "hover_feasible": hover_feasible,
                    "hover_best_gap": hover_gap,
                    "joint_margin": gap,
                    "measured_grip": False,
                    "carry_path_validated": False,
                }
            )
            if accepted:
                if on_heading_selected is not None:
                    on_heading_selected(heading)
                return pick
            if hover_feasible is False:
                break  # Changing release heading cannot make this grip's hover reachable.
    raise ValueError("No candidate grasp has a bounded seated-pose forecast")


class PregraspDescent:
    """Straight Cartesian ingress after measured hover arrival, with no body writes."""

    def __init__(self, motion, target, orientation, tcp_local):
        self.motion = motion
        self.target = np.asarray(target).copy()
        self.orientation = np.asarray(orientation).copy()
        self.rotation = Rotation.from_quat(self.orientation[[1, 2, 3, 0]])
        self.tcp_target = self.target + self.rotation.apply(tcp_local)
        self.steps = self.stable = 0

    def forward(self, position, matrix):
        self.steps += 1
        distance = float(np.linalg.norm(np.asarray(position) - self.target))
        angle = float((Rotation.from_matrix(matrix) * self.rotation.inv()).magnitude())
        self.stable = self.stable + 1 if distance < 0.002 and angle < np.radians(2) else 0
        if self.steps > 600:
            raise ValueError("Pregrasp descent exceeded bounded 600-step budget")
        action = self.motion.forward(
            target_end_effector_position=self.tcp_target, target_end_effector_orientation=self.orientation
        )
        return action, {
            "type": "pregrasp_descent_sample",
            "steps": self.steps,
            "position_error": distance,
            "orientation_error": angle,
            "reached": self.stable >= 10,
        }


def open_hand_planning_scene(
    active, state, rack_checker, usd, stage, output, *, measured_hand=False, include_target=False
):
    """Include real open fingers and bench/other tools; target tool stays explicit.

    The target pipette is omitted from this coarse planning representation
    because its global box falsely overlaps the enclosing hand at the grasp.
    Executed approach must retain the physical contact/object-motion aborts.
    """
    joints = active.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
    position, matrix = active.rmp_controller.rmp_flow.get_end_effector_pose(joints)
    descriptor = yaml.safe_load(Path(active.rmp_controller.rmp_flow_config["robot_description_path"]).read_text())
    descriptor["collision_spheres"] = [g for g in descriptor["collision_spheres"] if "panda_link0" not in g]
    additions, hand_geometry = [], []
    links = (["panda_hand"] if measured_hand else []) + ["panda_leftfinger", "panda_rightfinger"]
    for finger in links:
        root = active.robot.prim_path_str + "/" + finger
        groups = (
            world_collision_meshes(usd, stage, root)
            if finger == "panda_hand"
            else {root: world_mesh_vertices(usd, stage, root)}
        )
        for path, vertices in groups.items():
            points = (vertices - position) @ matrix
            box, frame = oriented_mesh_bounds(points)
            centers, radii = (
                cover_convex_mesh_with_spheres(points) if finger == "panda_hand" else cover_box_with_spheres(box, frame)
            )
            additions.extend({"center": c.tolist(), "radius": float(r)} for c, r in zip(centers, radii))
            hand_geometry.append(
                {
                    "path": path,
                    "vertices": len(points),
                    "spheres": len(centers),
                    "eef_bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                }
            )
    descriptor["collision_spheres"].append({active.rmp_controller.rmp_flow.end_effector_frame_name: additions})
    generated_path = Path(output) / f"approach_descriptor_{state['object_path'].rsplit('/', 1)[-1]}.yaml"
    with generated_path.open("x") as generated:
        yaml.safe_dump(descriptor, generated)
    boxes = rack_checker.boxes.tolist()
    frames = (
        rack_checker.box_frames if rack_checker.box_frames is not None else np.tile(np.eye(4), (len(boxes), 1, 1))
    ).tolist()
    paths = [
        "/World/Cube",
        *[
            obj["object_path"]
            for obj in state.get("pipettes", [state])
            if include_target or obj["object_path"] != state["object_path"]
        ],
    ]
    cache = usd.UsdGeom.BBoxCache(0, ["default"])
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError(f"Missing approach obstacle: {path}")
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        boxes.append([list(bounds.GetMin()), list(bounds.GetMax())])
        frames.append(np.eye(4).tolist())
    checker = RackArmClearance.from_descriptor(
        rack_checker.solver, generated_path, boxes, margin=rack_checker.margin, box_frames=frames
    )
    return (
        generated_path,
        checker,
        {
            "type": "open_hand_planning_geometry",
            "hand_and_finger_spheres": len(additions),
            "hand_cover": "physical_mesh_convex_hulls" if measured_hand else "native_descriptor",
            "mesh_geometry": hand_geometry,
            "start_limiting_contact": checker.limiting_contact(joints),
            "obstacle_boxes": boxes,
            "obstacle_frames": frames,
            "extra_obstacles": paths,
            "target_excluded_from_coarse_planning": None if include_target else state["object_path"],
            "planning_base_exempt": "panda_link0",
            "physical_collisions_modified": False,
        },
    )


class CarryRRTProbe:
    def __init__(
        self,
        active,
        state,
        rack_checker,
        usd,
        stage,
        output,
        seed,
        emit,
        interpolate,
        *,
        measured_hand=False,
        optimize_pose=False,
        carry_margin=0.002,
        placement_robot_frames=None,
    ):
        if not np.isfinite(carry_margin) or not 0.002 <= carry_margin <= 0.02:
            raise ValueError("Carry planning margin must be finite and in [2, 20] mm")
        self.emit = emit
        self.index = self.steps = self.stable = 0
        self.done = False
        self.hold_steps = 0
        self.tracking_bias = np.zeros(7)
        self.frame_name = active.rmp_controller.rmp_flow.end_effector_frame_name
        self.tcp_local = active.place_controller._cspace_controller.tcp_local_offset.copy()
        self.orientation = np.asarray(active._placement_orientation)
        rotation = Rotation.from_quat(self.orientation[[1, 2, 3, 0]])
        self.tcp_target = (
            active._command_seat_position(state)
            + active._grasp_offset
            + np.array([-0.12, 0, active._release_clearance()])
        )
        self.target = self.tcp_target - rotation.apply(self.tcp_local)
        self.other_objects = {
            obj["object_path"]: np.asarray(obj["object_grasp_position"]).copy()
            for obj in state.get("pipettes", [state])
            if obj["object_path"] != state["object_path"]
        }
        start = np.asarray(active.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions())
        position, matrix = active.rmp_controller.rmp_flow.get_end_effector_pose(start)
        source = Path(active.rmp_controller.rmp_flow_config["robot_description_path"])
        descriptor = yaml.safe_load(source.read_text())
        # The bolted base is already in contact with the bench. Exempt ONLY this
        # immovable frame in planning, not any moving link or physical collider.
        descriptor["collision_spheres"] = [g for g in descriptor["collision_spheres"] if "panda_link0" not in g]
        additions, geometry = [], []
        links = (["panda_hand"] if measured_hand else []) + ["panda_leftfinger", "panda_rightfinger"]
        paths = [
            state["object_path"],
            *[active.robot.prim_path_str + "/" + finger for finger in links],
        ]
        meshes = {}
        for prim_path in paths:
            if prim_path.endswith("/panda_hand"):
                meshes.update(
                    {path: (vertices, True) for path, vertices in world_collision_meshes(usd, stage, prim_path).items()}
                )
            else:
                meshes[prim_path] = (world_mesh_vertices(usd, stage, prim_path), False)
        for prim_path, (points, hand_collider) in meshes.items():
            vertices = (points - position) @ matrix
            box, frame = oriented_mesh_bounds(vertices)
            centers, radii = (
                cover_convex_mesh_with_spheres(vertices)
                if hand_collider
                else cover_box_with_spheres(box, frame, cell_size=0.02)
            )
            additions.extend(
                {"center": center.tolist(), "radius": float(radius)} for center, radius in zip(centers, radii)
            )
            geometry.append(
                {
                    "prim_path": prim_path,
                    "vertices": len(vertices),
                    "spheres": len(centers),
                    "cover": "physical_mesh_convex_hull_cells" if hand_collider else "box_cells",
                    "bounds": box.tolist(),
                    "frame_in_eef": frame.tolist(),
                }
            )
        descriptor["collision_spheres"].append({self.frame_name: additions})
        descriptor_path = Path(output) / f"carry_descriptor_{state['object_path'].rsplit('/', 1)[-1]}.yaml"
        with descriptor_path.open("x") as generated:
            yaml.safe_dump(descriptor, generated)
        boxes = rack_checker.boxes.tolist()
        frames = (
            rack_checker.box_frames if rack_checker.box_frames is not None else np.tile(np.eye(4), (len(boxes), 1, 1))
        ).tolist()
        box_paths = [f"{state['target_path']}:part{i}" for i in range(len(boxes))]
        obstacle_paths = ["/World/Cube", *self.other_objects]
        cache = usd.UsdGeom.BBoxCache(0, ["default"])
        for prim_path in obstacle_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise ValueError(f"Missing carry obstacle: {prim_path}")
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            boxes.append([list(bounds.GetMin()), list(bounds.GetMax())])
            frames.append(np.eye(4).tolist())
            box_paths.append(prim_path)
        self.checker = RackArmClearance.from_descriptor(
            rack_checker.solver, descriptor_path, boxes, margin=carry_margin, box_frames=frames
        )
        placement_checker = None
        if placement_robot_frames is not None:
            # Reuse the physically measured OPEN-hand cover captured at this
            # tool's pregrasp, but rebuild obstacle boxes at the current scene
            # state. Thus already placed tools are not left at their old table
            # poses. Only the currently held tool is absent from these boxes.
            if not optimize_pose or "panda_link0" in placement_robot_frames:
                raise ValueError("Placement geometry needs endpoint optimization and moving-robot frames only")
            spheres = {
                name: [{"center": center, "radius": radius} for center, radius in zip(centers, radii)]
                for name, (centers, radii) in placement_robot_frames.items()
            }
            placement_checker = RackArmClearance(
                rack_checker.solver, spheres, boxes, margin=carry_margin, box_frames=frames
            )
            emit(
                {
                    "type": "placement_robot_geometry",
                    "open_hand_cover": True,
                    "obstacle_paths": box_paths,
                    "excluded_current_tool": state["object_path"],
                    "frames": list(spheres),
                    "margin": carry_margin,
                    "physical_collisions_modified": False,
                }
            )
        start_contact = self.checker.limiting_contact(start)
        emit(
            {
                "type": "carry_geometry",
                "planning_margin": carry_margin,
                "geometry": geometry,
                "hand_cover": "physical_mesh_convex_hulls" if measured_hand else "native_descriptor",
                "obstacle_paths": box_paths,
                "planning_base_exempt": "panda_link0",
                "physical_collisions_modified": False,
                "start_limiting": {**start_contact, "obstacle": box_paths[start_contact["box"]]},
                "target": self.target.tolist(),
                "orientation": self.orientation.tolist(),
            }
        )
        if start_contact["clearance"] < self.checker.margin:
            raise ValueError("Measured lifted pose is not clear under carried-object planning cover")
        planner_config = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
        for key in ("urdf_path", "end_effector_frame_name"):
            planner_config[key] = active.rmp_controller.rmp_flow_config[key]
        planner_config["robot_description_path"] = str(descriptor_path)
        planner = RRT(**planner_config)
        planner.set_robot_base_pose(*active.robot.get_world_pose())
        planner.set_max_iterations(4000)
        planner.set_param("step_size", 0.03)
        for box, frame in zip(boxes, frames):
            if not planner.add_cuboid(PlanningBox(box, frame, self.checker.margin), static=True):
                raise ValueError("Could not register carry planning obstacle")
        planner.update_world()
        planner.set_end_effector_target(self.target, self.orientation)
        limits = np.asarray(active.robot._articulation_view.get_dof_limits())[0, :7]
        self.joint_bounds = limits.copy()
        self.joint_margin = float(active.cfg.task.pipette_preflight_joint_margin)
        self.commands = None
        selected_path = None
        for attempt in range(4):
            planner.set_random_seed(seed + attempt)
            path = planner.compute_path(start, np.empty(0))
            record = {
                "type": "carry_rrt_plan",
                "seed": seed + attempt,
                "path_found": path is not None,
                "path_executed": False,
            }
            if path is not None:
                dense = [np.asarray(path[0])]
                for first, last in pairwise(path):
                    count = max(1, int(np.ceil(np.max(np.abs(last - first)) / 0.01)))
                    dense.extend(first + t * (last - first) for t in np.linspace(0, 1, count + 1)[1:])
                joint_margin = float(np.min(np.minimum(path - limits[:, 0], limits[:, 1] - path)))
                gap = min(self.checker.minimum_clearance(q) for q in dense)
                record.update(
                    path=np.asarray(path).tolist(), dense_samples=len(dense), minimum_gap=gap, joint_margin=joint_margin
                )
                if gap >= self.checker.margin and joint_margin >= float(active.cfg.task.pipette_preflight_joint_margin):
                    shortened = shortcut_joint_path(path, self.checker, limits, joint_margin=self.joint_margin)
                    commands = interpolate(shortened)
                    # Validate every actual interpolation command as well as
                    # the coarse planning samples. The fixture records its
                    # explicit interpolation rate; contact/arrival/time limits
                    # remain unchanged for every candidate rate.
                    command_gap = min(self.checker.minimum_clearance(q) for q in commands)
                    record.update(
                        shortened_path=shortened.tolist(),
                        original_waypoints=len(path),
                        shortened_waypoints=len(shortened),
                        execution_commands=len(commands),
                        minimum_command_gap=command_gap,
                    )
                    if command_gap >= self.checker.margin:
                        self.commands = commands
                        selected_path = shortened
            emit(record)
            if self.commands is not None:
                break
        if optimize_pose:
            # Use the ORIGINAL placement offsets, not a second approximate
            # insertion definition. Closed-loop alignment may still fail in
            # execution: this is a kinematic preflight, never a success claim.
            seated = active._command_seat_position(state) + active._grasp_offset + [0, 0, active._release_clearance()]
            retreat = active.place_controller.retreat_offset
            if getattr(active.cfg.task, "pipette_axial_retreat", False):
                retreat = rotation.apply([0, 0, -0.12])
            if not getattr(active.cfg.task, "pipette_front_insertion", False) or retreat is None:
                raise ValueError("Carry continuation forecast requires explicit front insertion and retreat offsets")
            tcp_targets = [seated + active.place_controller.approach_offset, seated, seated + retreat]
            if active.place_controller.post_retreat_position is not None:
                tcp_targets.append(active.place_controller.post_retreat_position)
            targets = np.asarray(tcp_targets) - rotation.apply(self.tcp_local)

            def validate_endpoint(joints):
                return screen_pose_continuation(
                    self.checker.solver,
                    continuous_inverse_kinematics,
                    self.frame_name,
                    joints,
                    targets,
                    self.orientation,
                    limits,
                    joint_margin=self.joint_margin,
                    geometry_checker=placement_checker,
                )

            selected_path, details = improve_pose_path(
                planner,
                self.checker,
                self.frame_name,
                self.target,
                self.orientation,
                start,
                selected_path,
                limits,
                interpolate,
                seed,
                joint_margin=self.joint_margin,
                endpoint_validator=validate_endpoint,
            )
            details["continuation_targets"] = targets.tolist()
            emit({"type": "carry_pose_path_optimization", "object_path": state["object_path"], **details})
            self.commands = None if selected_path is None else interpolate(selected_path)
        if self.commands is None:
            raise ValueError("No dense-checked global carry path with held-tool/finger clearance")

    def forward(self, active, state):
        self.steps += 1
        drift = active._trace_grasp(state)
        if drift is not None and drift > float(active.cfg.task.max_grasp_rotation_deg):
            raise ValueError(f"Carry slip exceeds original guard: {drift:.4f} deg")
        for obj in state.get("pipettes", [state]):
            if obj["object_path"] in self.other_objects and (
                np.linalg.norm(obj["object_grasp_position"] - self.other_objects[obj["object_path"]]) > 0.001
            ):
                raise ValueError("Another pipette moved during global carry")
        joints = np.asarray(active.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions())
        actual_gap = self.checker.minimum_clearance(joints)
        if actual_gap < 0:
            raise ValueError(f"Measured carry geometry lost clearance: {actual_gap:.6f} m")
        error = float(np.max(np.abs(joints - self.commands[self.index])))
        if error < 0.03 and self.index < len(self.commands) - 1:
            self.index += 1
            self.hold_steps = 0
        else:
            self.hold_steps += 1
        # Integrate only after holding the SAME path point, with modest tracking
        # error and measured clearance. Cap at the existing 0.03 rad envelope;
        # never integrate through a large error or physical-contact abort.
        if self.hold_steps >= 5 and error < 0.06:
            self.tracking_bias = np.clip(self.tracking_bias + 0.08 * (self.commands[self.index] - joints), -0.03, 0.03)
        _position, matrix = active.rmp_controller.rmp_flow.get_end_effector_pose(joints)
        angle = float(
            (Rotation.from_matrix(matrix) * Rotation.from_quat(self.orientation[[1, 2, 3, 0]]).inv()).magnitude()
        )
        distance = float(np.linalg.norm(state["gripper_position"] - self.tcp_target))
        reached = self.index == len(self.commands) - 1 and distance < 0.002 and angle < np.radians(2)
        self.stable = self.stable + 1 if reached else 0
        self.done = self.stable >= 10
        self.emit(
            {
                "type": "carry_rrt_sample",
                "steps": self.steps,
                "index": self.index,
                "tcp_error": distance,
                "orientation_error": angle,
                "tracking_error": error,
                "actual_minimum_gap": actual_gap,
                "tracking_bias": self.tracking_bias.tolist(),
                "reached": self.done,
            }
        )
        if self.steps > 3000:
            raise ValueError("Global carry tracking exceeded bounded 3000-step budget")
        command = np.clip(
            self.commands[self.index] + self.tracking_bias,
            self.joint_bounds[:, 0] + self.joint_margin,
            self.joint_bounds[:, 1] - self.joint_margin,
        )
        self.tracking_bias = command - self.commands[self.index]
        return ArticulationAction(joint_positions=command, joint_indices=np.arange(7))
