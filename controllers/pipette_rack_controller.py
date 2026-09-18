"""Pick a pipette and seat it in the empty slot adjacent to the reference one."""

import os
import random
from itertools import pairwise, product

import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from loguru import logger
from scipy.spatial.transform import Rotation as R

from .flask_to_cork_controller import FlaskToCorkTaskController, Phase
from .pipette_kinematics import continuous_inverse_kinematics


class PipettePoseMotion:
    """Slow, continuous joint commands for preflighted placement waypoints.

    Seed the high approach from the seated IK branch, so a locally attractive
    but joint-limited wrist posture is not selected during transfer.
    """

    def __init__(
        self,
        rmp_controller,
        seed,
        tcp_local_offset=None,
        cartesian_first=False,
        transfer_time_scale=1.0,
        local_ik_recovery=False,
        joint_bounds=None,
        joint_margin=0.05,
    ):
        self.rmp = rmp_controller
        self.solver = rmp_controller.rmp_flow.get_kinematics_solver()
        self.seed = np.asarray(seed)
        self.goal = None
        self.step = 0
        self.tracking_bias = np.zeros(7)
        self.tracking_error_limit = 0.03
        self.lateral_error = None
        self.lateral_integral = np.zeros(2)
        self.angular_error = None
        self.angular_integral = np.zeros(3)
        self.tcp_local_offset = np.zeros(3) if tcp_local_offset is None else np.asarray(tcp_local_offset)
        self.cartesian_first = bool(cartesian_first)
        self.transfer_time_scale = float(transfer_time_scale)
        self.local_ik_recovery = bool(local_ik_recovery)
        self.joint_bounds = None if joint_bounds is None else np.asarray(joint_bounds, dtype=float).copy()
        self.joint_margin = float(joint_margin)
        if not np.isfinite(self.joint_margin) or not 0.05 <= self.joint_margin <= 0.25:
            raise ValueError("Pipette runtime joint margin must be in [0.05, 0.25] radians")
        if self.joint_bounds is not None and (
            self.joint_bounds.shape != (7, 2)
            or not np.isfinite(self.joint_bounds).all()
            or np.any(self.joint_bounds[:, 1] - self.joint_bounds[:, 0] <= 2 * self.joint_margin)
        ):
            raise ValueError("Expected seven finite, nonempty physical joint intervals")
        if not np.isfinite(self.transfer_time_scale) or self.transfer_time_scale < 1:
            raise ValueError("Pipette transfer time scale must be finite and at least 1")

    def forward(self, target_end_effector_position, target_end_effector_orientation):
        orientation = R.from_quat(np.asarray(target_end_effector_orientation)[[1, 2, 3, 0]])
        target_end_effector_position = np.asarray(target_end_effector_position) - orientation.apply(
            self.tcp_local_offset
        )
        goal = np.r_[target_end_effector_position, target_end_effector_orientation]
        if self.goal is None or not np.allclose(goal, self.goal, atol=1e-6):
            self.cartesian = self.goal is not None or self.cartesian_first
            if not self.cartesian or self.joint_bounds is None:
                # A joint-space move needs an endpoint solution. Keep the
                # historical endpoint probe for legacy unbounded configs too.
                solution, ok = self.solver.compute_inverse_kinematics(
                    self.rmp.rmp_flow.end_effector_frame_name,
                    np.asarray(target_end_effector_position),
                    np.asarray(target_end_effector_orientation),
                    warm_start=self.seed,
                    position_tolerance=0.001,
                    orientation_tolerance=0.01,
                )
                if self.joint_bounds is not None and ok and not self._inside_joint_margin(solution):
                    solution, ok = continuous_inverse_kinematics(
                        self.solver,
                        self.rmp.rmp_flow.end_effector_frame_name,
                        target_end_effector_position,
                        target_end_effector_orientation,
                        np.asarray(solution),
                        self.joint_bounds,
                        joint_margin=self.joint_margin,
                    )
                if not ok or not self._inside_joint_margin(solution):
                    raise ValueError("Pipette placement waypoint is not IK reachable")
                self.seed = np.asarray(solution)
            self.start = self.rmp.articulation_rmp.get_active_joints_subset().get_joint_positions().copy()
            self.start_position, matrix = self.rmp.rmp_flow.get_end_effector_pose(self.start)
            self.start_rotation = R.from_matrix(matrix)
            self.rotation_delta = (
                R.from_quat(np.asarray(target_end_effector_orientation)[[1, 2, 3, 0]]) * self.start_rotation.inv()
            ).as_rotvec()
            if not self.cartesian:
                duration = max(80, int(np.ceil(np.max(np.abs(self.seed - self.start)) * 120)))
                self.duration = int(np.ceil(duration * self.transfer_time_scale))
            if self.cartesian:
                # Continue from the measured branch. A discarded endpoint IK
                # on another redundant branch cannot veto this local path.
                # Every command, including arrival, still receives bounded IK.
                duration = max(
                    80,
                    int(np.ceil(np.linalg.norm(target_end_effector_position - self.start_position) / 0.0009)),
                    int(np.ceil(np.linalg.norm(self.rotation_delta) / 0.004)),
                )
                self.duration = int(np.ceil(duration * self.transfer_time_scale))
                self.seed = self.start.copy()
            self.goal, self.step = goal, 0
        self.step += 1
        t = min(self.step / self.duration, 1.0)
        blend = t**3 * (10 - 15 * t + 6 * t**2)
        if self.cartesian:
            position = self.start_position + blend * (self.goal[:3] - self.start_position)
            if self.lateral_error is not None:
                self.lateral_integral = np.clip(self.lateral_integral + 0.05 * self.lateral_error, -0.006, 0.006)
                position[:2] += 0.5 * self.lateral_error + self.lateral_integral
            rotation = R.from_rotvec(blend * self.rotation_delta) * self.start_rotation
            if self.angular_error is not None:
                self.angular_integral = np.clip(self.angular_integral + 0.025 * self.angular_error, -0.35, 0.35)
                rotation = R.from_rotvec(0.5 * self.angular_error + self.angular_integral) * rotation
            quat = rotation.as_quat()[[3, 0, 1, 2]]
            solution, ok = self.solver.compute_inverse_kinematics(
                self.rmp.rmp_flow.end_effector_frame_name,
                position,
                quat,
                warm_start=self.seed,
                position_tolerance=0.0005,
                orientation_tolerance=0.005,
            )
            if (self.local_ik_recovery or self.joint_bounds is not None) and (
                not ok or np.max(np.abs(solution - self.seed)) > 0.35 or not self._inside_joint_margin(solution)
            ):
                solution, ok = continuous_inverse_kinematics(
                    self.solver,
                    self.rmp.rmp_flow.end_effector_frame_name,
                    position,
                    quat,
                    self.seed,
                    self.joint_bounds,
                    joint_margin=self.joint_margin,
                )
            if not ok or not self._inside_joint_margin(solution):
                raise ValueError("Pipette Cartesian segment has no continuous IK solution")
            if os.environ.get("LABUTOPIA_PIPETTE_GRASP_DEBUG") == "1" and np.max(np.abs(solution - self.seed)) > 0.03:
                logger.warning(
                    "Pipette IK step jump: step={}, delta={}, position={}", self.step, solution - self.seed, position
                )
            self.seed = np.asarray(solution)
            target = self.seed
        else:
            target = self.start + blend * (self.seed - self.start)
        if t == 1:
            measured = self.rmp.articulation_rmp.get_active_joints_subset().get_joint_positions()
            error = target - measured
            # Position-only drives retain a small gravity-load error. Integrate
            # only at a stationary waypoint; never wind up against an obstacle.
            if np.max(np.abs(error)) < self.tracking_error_limit:
                self.tracking_bias = np.clip(self.tracking_bias + 0.08 * error, -0.03, 0.03)
        command = target + self.tracking_bias
        if self.joint_bounds is not None:
            # Gravity compensation must not spend the margin protected by IK.
            command = np.clip(
                command, self.joint_bounds[:, 0] + self.joint_margin, self.joint_bounds[:, 1] - self.joint_margin
            )
            self.tracking_bias = command - target
        return ArticulationAction(joint_positions=command, joint_indices=np.arange(7))

    def _inside_joint_margin(self, joints):
        if self.joint_bounds is None:
            return True
        joints = np.asarray(joints)
        return bool(
            joints.shape == (7,)
            and np.isfinite(joints).all()
            and np.all(joints >= self.joint_bounds[:, 0] + self.joint_margin)
            and np.all(joints <= self.joint_bounds[:, 1] - self.joint_margin)
        )

    def reset(self):
        self.rmp.reset()
        self.goal = None
        self.step = 0
        self.tracking_bias.fill(0)
        self.lateral_integral.fill(0)
        self.angular_integral.fill(0)
        self.lateral_error = None
        self.angular_error = None


class PipetteRackTaskController(FlaskToCorkTaskController):
    """Use the rack's seating geometry and rotate the measured grasp offset."""

    def _placement_feedback(self, state):
        """Keep insertion on its centreline without accelerating axial travel.

        Experimental feedback affects robot commands only. The path generator
        retains control of motion along the slot; release keeps its original
        full-pose checks. No body pose or physical contact is changed.
        """
        event = self.place_controller._event
        inserting = (
            event == 0
            and bool(getattr(self.cfg.task, "pipette_track_insertion_axis", False))
            and bool(getattr(self.cfg.task, "pipette_front_insertion", False))
            and self.place_controller._approach_complete
        )
        if event != 1 and not inserting:
            return None, None
        lateral = (self._command_seat_position(state) - state["object_reference_position"])[:2]
        if inserting:
            direction = np.asarray(self.place_controller.approach_offset, dtype=float)[:2]
            length = np.linalg.norm(direction)
            if not np.isfinite(length) or length < 1e-6:
                raise ValueError("Horizontal insertion feedback requires a finite nonzero approach direction")
            direction = direction / length
            lateral = lateral - direction * np.dot(direction, lateral)
        correction = R.from_quat(state["target_quaternion"]) * R.from_quat(state["object_quaternion"]).inv()
        return lateral, correction.as_rotvec()

    def _motion_constraints(self):
        if not getattr(self.cfg.task, "pipette_enforce_runtime_joint_margin", False):
            return {}
        return {
            "joint_bounds": np.asarray(self.robot._articulation_view.get_dof_limits())[0, :7],
            "joint_margin": float(getattr(self.cfg.task, "pipette_preflight_joint_margin", 0.05)),
        }

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        self.pick_controller.arrival_stable_steps = 3
        self.pick_controller._position_threshold = 0.015
        self.pick_controller.grasp_position_threshold = 0.010
        self.pick_controller.hold_pose_during_close = bool(getattr(cfg.task, "pipette_hold_during_close", False))
        if getattr(cfg.task, "pipette_ik_pick", False):
            self.pick_controller._events_dt[0] = 1 / 900
        # The collar overhang provides 10 mm of lateral seating margin.
        self.place_controller.precision_position_threshold = 0.008
        self.place_controller.precision_height_threshold = 0.020
        self.place_controller.retreat_position_threshold = 0.025
        transfer_scale = float(getattr(cfg.task, "pipette_transfer_time_scale", 1.0))
        if not np.isfinite(transfer_scale) or transfer_scale < 1:
            raise ValueError("Pipette transfer time scale must be finite and at least 1")
        # Motion timeouts scale with Cartesian paths; settle/release timing
        # and all measured arrival/success thresholds remain unchanged.
        self.place_controller._events_dt[1] /= transfer_scale
        self.place_controller._events_dt[4] /= transfer_scale
        if getattr(cfg.task, "pipette_return_to_transit", False):
            self.place_controller.post_retreat_position = np.asarray(cfg.task.pipette_transit_position)
            self.place_controller._events_dt[4] = 1 / (900 * transfer_scale)
        self.place_controller.approach_offset = np.array([-0.14, 0.0, 0.0])
        if getattr(cfg.task, "reference_geometry", False):
            # A continuous IK branch change can exceed the generic 400-frame
            # transfer budget. This changes motion time, not success tolerance.
            self.place_controller._events_dt[0] = 1 / (
                800 * float(getattr(cfg.task, "pipette_transfer_time_scale", 1.0))
            )
            self.pick_controller.grasp_position_threshold = 0.005
            if getattr(robot, "_gripper_control_mode", "position") == "force":
                self.pick_controller.close_contact_gap_range = (0.016, 0.036)
                if getattr(cfg.task, "gripper_force_servo_mode", "velocity") == "position":
                    self.pick_controller.close_stable_steps = 30
            self.place_controller.precision_position_threshold = 0.0015
            self.place_controller.precision_height_threshold = float(
                getattr(cfg.task, "pipette_release_height_tolerance", 0.004)
            )
            self.place_controller.approach_offset = np.array([-0.08, 0.0, 0.0])
            if getattr(cfg.task, "pipette_front_insertion", False):
                self.place_controller.approach_offset = np.array([-0.12, 0.0, 0.0])
        self._placement_offset_refined = False
        self._diagnostic_steps = 0
        self._alignment_steps = 0
        self._alignment_done = False
        self._grasp_reference = None
        self._grasp_trace_steps = 0
        self._lift_motion_configured = False
        self._pick_target_orientation = None
        self._close_previous_pose = None
        self._planned_pick_orientation = None
        self._preflight_wait_steps = 0
        self._transit_phase = self._transit_steps = self._transit_stable = 0

    def reset(self):
        super().reset()
        self._preflight_wait_steps = 0
        self._transit_phase = self._transit_steps = self._transit_stable = 0
        self._placement_offset_refined = False
        self._diagnostic_steps = 0
        self._alignment_steps = 0
        self._alignment_done = False
        self._grasp_reference = None
        self._grasp_trace_steps = 0
        self._lift_motion_configured = False
        self._pick_target_orientation = None
        self._close_previous_pose = None
        self._planned_pick_orientation = None
        if self.mode == "collect":
            self.place_controller._cspace_controller = self.rmp_controller
            self.pick_controller._cspace_controller = self.rmp_controller

    def _sample_phase_instruction(self, phase):
        if phase == Phase.PICKING:
            templates = self._build_instruction_templates(
                "Pick up the pipette from the table",
                "Pick up the pipette from the table and lift it",
            )
        else:
            templates = self._build_instruction_templates(
                "Place the pipette onto the rack",
                "Move the pipette to the empty rack slot and place it down",
            )
        return random.choice(templates)

    @staticmethod
    def _orientation(pitch):
        return R.from_euler("xyz", np.radians([0, pitch, 0])).as_quat()

    def _pick_orientation(self, *, settled_state=None):
        if getattr(self, "_planned_pick_orientation", None) is not None:
            return self._planned_pick_orientation.copy()
        orientation = self._orientation(float(getattr(self.cfg.task, "pipette_pick_pitch", 170)))
        yaw = float(getattr(self.cfg.task, "pipette_pick_yaw", 0.0))
        # Preserve the established scalar-first controller convention, including
        # the legacy pitch mapping, while allowing a symmetric jaw approach.
        if yaw:
            hand = R.from_quat(orientation[[1, 2, 3, 0]])
            orientation = (R.from_euler("z", yaw, degrees=True) * hand).as_quat()[[3, 0, 1, 2]]
        if getattr(self.cfg.task, "pipette_object_aligned_grasp", False) and (
            settled_state is not None or self.pick_controller._event >= 1
        ):
            if self._pick_target_orientation is None:
                # The authored horizontal tool is Ry(90). Follow its measured
                # roll/yaw once it has settled, then lock the grasp orientation;
                # never rotate the target in response to subsequent grip slip.
                source_state = self.state if settled_state is None else settled_state
                correction = R.from_quat(source_state["object_quaternion"]) * R.from_euler("y", -90, degrees=True)
                hand = R.from_quat(orientation[[1, 2, 3, 0]])
                self._pick_target_orientation = (correction * hand).as_quat()[[3, 0, 1, 2]]
            orientation = self._pick_target_orientation.copy()
        return orientation

    def _plan_grasp_branch(self, state, tcp_local):
        """Backtrack a continuous carry path before closing the fingers.

        Reconfigure the redundant arm while EMPTY, rather than changing wrist
        branches with a pipette in the hand. This is an IK feasibility check,
        not a collision bypass; normal contacts and all success gates remain.
        """
        solver = self.rmp_controller.rmp_flow.get_kinematics_solver()
        frame = self.rmp_controller.rmp_flow.end_effector_frame_name
        actual = self.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
        bounds = np.asarray(self.robot._articulation_view.get_dof_limits())[0, :7]
        joint_margin = float(getattr(self.cfg.task, "pipette_preflight_joint_margin", 0.05))
        if not np.isfinite(joint_margin) or not 0.05 <= joint_margin <= 0.25:
            raise ValueError("Pipette preflight joint margin must be in [0.05, 0.25] radians")
        grasp = np.asarray(state["object_grasp_position"]) + np.array(
            [0, 0, float(getattr(self.cfg.task, "pipette_grasp_height_offset", -0.003))]
        )
        lift_height = float(getattr(self.cfg.task, "pipette_lift_clearance", 0.30)) + 0.04
        source = R.from_quat(state["object_quaternion"])
        correction = R.from_quat(state["target_quaternion"]) * source.inv()
        # Preflight runs at event 0, after the explicit scene-settle gate.
        # Align here before caching a plan; otherwise the cached nominal pose
        # bypasses object alignment on every subsequent pickup step.
        base = R.from_quat(self._pick_orientation(settled_state=state)[[1, 2, 3, 0]])
        warm_starts = (
            actual,
            np.array([-1, -0.7, 1.4, -2.2, 2, 2.4, 0]),
            np.array([1, -0.7, -1.4, -2.2, -2, 2.4, 0]),
            *(
                np.array([shoulder, -0.7, -shoulder, -2.2, wrist, 2.4, roll])
                for shoulder, wrist, roll in product((-1.5, 0.0, 1.5), (-2.0, 2.0), (-2.0, 2.0))
            ),
        )
        candidates = []
        failures = {key: 0 for key in ("initial_ik", "initial_margin", "path_ik", "path_jump", "path_margin")}
        # The near-COM housing is below the asymmetric thumb hook. Either
        # symmetric jaw assignment can grasp it, offering different reachable
        # arm branches without changing the desired pipette orientation.
        hands = (base,)
        if getattr(self.cfg.task, "pipette_symmetric_grasp_search", False):
            hands = (base, R.from_euler("z", 180, degrees=True) * base)
        # Rotate about the jaw-closing axis, retaining the same opposing pad
        # normals. Tilt candidates are only tried if the nominal grasp has no
        # continuous path; the selected orientation is locked before pickup.
        hands = tuple(
            (R.from_rotvec(hand.apply([0, np.radians(tilt), 0])) * hand, tilt)
            for tilt in getattr(self.cfg.task, "pipette_grasp_pitch_candidates_deg", [0.0])
            for hand in hands
        )
        previous_tilt = None
        for hand, tilt in hands:
            if candidates and tilt != previous_tilt:
                break
            previous_tilt = tilt
            placed_hand = correction * hand
            placed = self._command_seat_position(state) + correction.apply(grasp - state["object_reference_position"])
            waypoints = (
                (placed + np.array([0, 0, self._release_clearance()]), placed_hand),
                (placed + np.array([0, 0, 0.22]), placed_hand),
                (placed + np.array([-0.08, 0, 0.22]), placed_hand),
                (grasp + np.array([0, 0, lift_height]), hand),
                (grasp, hand),
            )
            if getattr(self.cfg.task, "pipette_front_insertion", False):
                release = placed + np.array([0, 0, self._release_clearance()])
                waypoints = (
                    (release, placed_hand),
                    (release + np.array([-0.12, 0, 0]), placed_hand),
                    (grasp + np.array([0, 0, lift_height]), hand),
                    (grasp, hand),
                )
                transit = getattr(self.cfg.task, "pipette_transit_position", None)
                if transit is not None:
                    hub = np.asarray(transit)
                    waypoints = (
                        waypoints[0],
                        waypoints[1],
                        (hub, placed_hand),
                        (hub, hand),
                        waypoints[2],
                        waypoints[3],
                    )
            for backwards, warm in product((True, False), warm_starts):
                route = waypoints if backwards else tuple(reversed(waypoints))
                first_position, first_rotation = route[0]
                seed, ok = solver.compute_inverse_kinematics(
                    frame,
                    first_position - first_rotation.apply(tcp_local),
                    first_rotation.as_quat()[[3, 0, 1, 2]],
                    warm_start=warm,
                    position_tolerance=0.0005,
                    orientation_tolerance=0.005,
                )
                if not ok:
                    failures["initial_ik"] += 1
                    continue
                initial_margin = float(np.min(np.minimum(seed - bounds[:, 0], bounds[:, 1] - seed)))
                if initial_margin < joint_margin:
                    failures["initial_margin"] += 1
                    continue
                grasp_seed = np.asarray(seed).copy()
                cost, margin = 0.0, initial_margin
                limiting = (first_position.copy(), np.asarray(seed).copy())
                for (start, rotation), (end, end_rotation) in pairwise(route):
                    delta = (end_rotation * rotation.inv()).as_rotvec()
                    count = max(
                        2,
                        int(np.ceil(np.linalg.norm(end - start) / 0.005)),
                        int(np.ceil(np.linalg.norm(delta) / 0.025)),
                    )
                    for t in np.linspace(0, 1, count + 1)[1:]:
                        orient = R.from_rotvec(t * delta) * rotation
                        position = start + t * (end - start) - orient.apply(tcp_local)
                        solution, ok = solver.compute_inverse_kinematics(
                            frame,
                            position,
                            orient.as_quat()[[3, 0, 1, 2]],
                            warm_start=seed,
                            position_tolerance=0.0005,
                            orientation_tolerance=0.005,
                        )
                        if getattr(self.cfg.task, "pipette_local_ik_recovery", False) and (
                            not ok
                            or np.max(np.abs(solution - seed)) > 0.35
                            or np.min(np.minimum(solution - bounds[:, 0], bounds[:, 1] - solution)) < joint_margin
                        ):
                            solution, ok = continuous_inverse_kinematics(
                                solver,
                                frame,
                                position,
                                orient.as_quat()[[3, 0, 1, 2]],
                                seed,
                                bounds,
                                joint_margin=joint_margin,
                            )
                        if not ok or np.max(np.abs(solution - seed)) > 0.35:
                            if failures["path_ik"] + failures["path_jump"] < 3:
                                logger.debug(
                                    "Pipette preflight rejection: backwards={}, from={}, to={}, at={}, delta={}, IK={}",
                                    backwards,
                                    start,
                                    end,
                                    position,
                                    solution - seed,
                                    ok,
                                )
                            failures["path_ik" if not ok else "path_jump"] += 1
                            ok = False
                            break
                        clearance = float(np.min(np.minimum(solution - bounds[:, 0], bounds[:, 1] - solution)))
                        if clearance < joint_margin:
                            failures["path_margin"] += 1
                            ok = False
                            break
                        if clearance < margin:
                            margin = clearance
                            limiting = (position.copy(), np.asarray(solution).copy())
                        cost += float(np.sum((solution - seed) ** 2))
                        seed = np.asarray(solution)
                    if not ok:
                        break
                if ok:
                    if backwards:
                        grasp_seed = seed.copy()
                    cost += 0.05 * float(np.sum((grasp_seed - actual) ** 2)) + 0.005 / margin
                    candidates.append((cost, grasp_seed, hand.as_quat()[[3, 0, 1, 2]], margin, limiting))
        if not candidates:
            raise ValueError(f"No continuous grasp-to-seat IK branch with joint-limit clearance: {failures}")
        cost, seed, orientation, margin, limiting = min(candidates, key=lambda candidate: candidate[0])
        self._planned_pick_orientation = orientation
        logger.info(
            "Pipette preflight: {} feasible branches, selected cost={:.3f}, joint margin={:.3f} rad",
            len(candidates),
            cost,
            margin,
        )
        logger.debug("Pipette preflight limiting pose={}, joints={}", *limiting)
        return seed

    def _release_clearance(self) -> float:
        return float(getattr(self.cfg.task, "pipette_release_clearance", 0.014))

    def _command_seat_position(self, state):
        """Command depth compensation without moving the success-test target."""
        position = np.asarray(self._goal_position(state)).copy()
        depth = float(getattr(self.cfg.task, "pipette_insertion_depth_offset", 0.0))
        rotation = R.from_quat(state.get("target_quaternion", [0, 0, 0, 1]))
        return position + rotation.apply(np.array([depth, 0.0, 0.0]))

    def _capture_grasp_offset(self, state):
        # Match the actual pre-hung reference pose, including tilt acquired
        # during pickup, rather than assuming a fixed 50-degree turn.
        source_axis = np.asarray(state["object_axis"])
        target_axis = np.asarray(state["target_axis"])
        source_axis = source_axis / np.linalg.norm(source_axis)
        target_axis = target_axis / np.linalg.norm(target_axis)
        if "object_quaternion" in state:
            # The real finger rest is asymmetric: match the whole target pose.
            rotation = R.from_quat(state["target_quaternion"]) * R.from_quat(state["object_quaternion"]).inv()
        else:
            rotation, _ = R.align_vectors([target_axis], [source_axis])
            rotation = R.from_rotvec(target_axis * np.radians(30)) * rotation
        if "object_reference_position" in state:
            joints = self.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
            position, matrix = self.rmp_controller.rmp_flow.get_end_effector_pose(joints)
            hand = R.from_matrix(matrix)
            logger.debug(
                "Pipette measured hand: FK={}, TCP={}, quat_xyzw={}",
                position,
                state["gripper_position"],
                hand.as_quat(),
            )
        else:
            hand = R.from_quat(self._orientation(170)[[1, 2, 3, 0]])
        # Isaac RMPFlow consumes scalar-first (wxyz), not scipy's xyzw.
        self._placement_orientation = (rotation * hand).as_quat()[[3, 0, 1, 2]]
        self._transit_orientation = hand.as_quat()[[3, 0, 1, 2]]
        logger.debug(
            "Pipette transfer: source_axis={}, target_axis={}, hand_wxyz={}",
            source_axis,
            target_axis,
            self._placement_orientation,
        )
        offset = np.array(state["gripper_position"]) - state.get("object_reference_position", state["object_position"])
        if "object_reference_position" in state:
            solver = self.rmp_controller.rmp_flow.get_kinematics_solver()
            frame = self.rmp_controller.rmp_flow.end_effector_frame_name
            goal = self._command_seat_position(state) + rotation.apply(offset) + [0, 0, self._release_clearance()]
            tcp_local = hand.inv().apply(state["gripper_position"] - position)
            # IK consumes its FK frame, not the measured TCP. Match the same
            # offset conversion used by PipettePoseMotion.forward at runtime.
            fk_goal = goal - (rotation * hand).apply(tcp_local)
            solution, ok = solver.compute_inverse_kinematics(
                frame,
                fk_goal,
                self._placement_orientation,
                warm_start=joints,
                position_tolerance=0.001,
                orientation_tolerance=0.01,
            )
            logger.debug("Pipette seated IK: reachable={}, goal={}, joints={}", ok, goal, solution)
            if ok:
                self.place_controller._cspace_controller = PipettePoseMotion(
                    self.rmp_controller,
                    solution,
                    tcp_local,
                    cartesian_first=bool(getattr(self.cfg.task, "pipette_cartesian_transfer", False)),
                    transfer_time_scale=float(getattr(self.cfg.task, "pipette_transfer_time_scale", 1.0)),
                    local_ik_recovery=bool(getattr(self.cfg.task, "pipette_local_ik_recovery", False)),
                    **self._motion_constraints(),
                )
            else:
                self.place_controller._phase_failure_reason = "Pipette seated pose is not IK reachable"
                self.place_controller._event = len(self.place_controller._events_dt)
        return rotation.apply(offset)

    def _placement_orientation_ok(self, state) -> bool:
        source = np.asarray(state["object_axis"])
        target = np.asarray(state["target_axis"])
        cosine = float(np.dot(source, target) / max(np.linalg.norm(source) * np.linalg.norm(target), 1e-9))
        initial_size = getattr(self, "initial_size", None)
        required_extent = 0.8 * max(initial_size) if initial_size is not None else 0.25
        return bool(state["object_size"][2] > required_extent and cosine > np.cos(np.radians(20)))

    def _check_placed_on_target(self) -> bool:
        placed = super()._check_placed_on_target()
        if getattr(self, "mode", "") == "replay" and os.environ.get("LABUTOPIA_PIPETTE_GRASP_DEBUG") == "1":
            self._replay_pose_debug_steps = getattr(self, "_replay_pose_debug_steps", 0) + 1
            if self._replay_pose_debug_steps % 120 == 0:
                logger.info(
                    "Pipette replay pose: reference={}, quaternion={}, axis={}, TCP={}",
                    self.state.get("object_reference_position"),
                    self.state.get("object_quaternion"),
                    self.state.get("object_axis"),
                    self.state["gripper_position"],
                )
        grasp = self.state.get("object_grasp_position")
        # With a head grasp the TCP is already >70 mm from the body's
        # centre while holding it. Require clearance from the actual grasp
        # feature too, especially when replay checks every recorded step.
        if (
            grasp is not None
            and "object_reference_position" in self.state
            and np.linalg.norm(self.state["gripper_position"] - grasp) <= 0.07
        ):
            self._last_failure_reason = "Pipette gripper has not cleared the grasp feature"
            return False
        return placed

    def _placement_xy_ok(self, position, target) -> bool:
        # A fork supports a range of insertion depths along X, but lateral
        # alignment must keep the collar over the selected pair of tines.
        delta = np.abs(position[:2] - target[:2])
        x_tolerance = float(getattr(self.cfg.task, "placement_x_tolerance", 0.045))
        y_tolerance = float(getattr(self.cfg.task, "placement_y_tolerance", 0.020))
        return bool(delta[0] < x_tolerance and delta[1] < y_tolerance)

    def _forward_active(self, state):
        drift = self._trace_grasp(state)
        limit = float(getattr(self.cfg.task, "max_grasp_rotation_deg", 8.0))
        if drift is not None and drift > limit:
            reason = f"Pipette slipped inside closed fingers: relative rotation={drift:.2f} deg exceeds {limit:.2f} deg"
            if self.current_phase == Phase.PLACING:
                return self._placement_failure(state, reason)
            self.pick_controller._phase_failure_reason = reason
            self.pick_controller._is_done = True
            action = self.pick_controller._null_action(len(state["joint_positions"]))
            return action, self.pick_controller._build_record_array(action, state["joint_positions"])
        if self.current_phase == Phase.PICKING:
            if getattr(self.cfg.task, "pipette_planned_grasp_branch", False) and self._preflight_wait_steps < int(
                getattr(self.cfg.task, "pipette_scene_settle_steps", 0)
            ):
                self._preflight_wait_steps += 1
                self.pick_controller._open_gripper()
                action = self.pick_controller._null_action(len(state["joint_positions"]))
                return action, self.pick_controller._build_record_array(
                    action, state["joint_positions"], gripper_state=self.pick_controller._current_gripper_state
                )
            if getattr(self.cfg.task, "pipette_ik_pick", False) and not isinstance(
                self.pick_controller._cspace_controller, PipettePoseMotion
            ):
                joints = self.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
                position, matrix = self.rmp_controller.rmp_flow.get_end_effector_pose(joints)
                tcp_local = R.from_matrix(matrix).inv().apply(state["gripper_position"] - position)
                if getattr(self.cfg.task, "pipette_planned_grasp_branch", False):
                    try:
                        joints = self._plan_grasp_branch(state, tcp_local)
                    except ValueError as error:
                        self.pick_controller._phase_failure_reason = str(error)
                        self.pick_controller._is_done = True
                        action = self.pick_controller._null_action(len(state["joint_positions"]))
                        return action, self.pick_controller._build_record_array(action, state["joint_positions"])
                self.pick_controller._cspace_controller = PipettePoseMotion(
                    self.rmp_controller,
                    joints,
                    tcp_local,
                    local_ik_recovery=bool(getattr(self.cfg.task, "pipette_local_ik_recovery", False)),
                    **self._motion_constraints(),
                )
            if (
                self.pick_controller._event == 5
                and not self._lift_motion_configured
                and getattr(self.cfg.task, "reference_geometry", False)
            ):
                joints = self.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
                position, matrix = self.rmp_controller.rmp_flow.get_end_effector_pose(joints)
                tcp_local = R.from_matrix(matrix).inv().apply(state["gripper_position"] - position)
                if not isinstance(self.pick_controller._cspace_controller, PipettePoseMotion):
                    self.pick_controller._cspace_controller = PipettePoseMotion(
                        self.rmp_controller, joints, tcp_local, **self._motion_constraints()
                    )
                self._lift_motion_configured = True
            pick_position = np.array(state["object_grasp_position"], copy=True)
            if self.pick_controller._event == 4 and getattr(self.cfg.task, "pipette_stable_close", False):
                sample = (
                    np.asarray(state["object_grasp_position"]).copy(),
                    R.from_quat(state["object_quaternion"]),
                    np.asarray(state["gripper_position"]).copy(),
                )
                previous = self._close_previous_pose
                self.pick_controller.close_contact_stable = previous is not None and (
                    np.linalg.norm(sample[0] - previous[0]) < 0.0001
                    and (sample[1] * previous[1].inv()).magnitude() < np.radians(0.05)
                    and np.linalg.norm(sample[2] - previous[2]) < 0.0001
                )
                self._close_previous_pose = sample
            motion = self.pick_controller._cspace_controller
            if isinstance(motion, PipettePoseMotion):
                # Compensate the changed load while closing at a fixed pose.
                # During transport/insertion retain the tighter anti-windup
                # limit so an obstacle is not mistaken for gravity error.
                holding = self.pick_controller.hold_pose_during_close and self.pick_controller._event in (3, 4)
                motion.tracking_error_limit = 0.05 if holding else 0.03
            pick_position[2] += (
                float(getattr(self.cfg.task, "pipette_grasp_height_offset", -0.003))
                if getattr(self.cfg.task, "reference_geometry", False)
                else 0.004
            )
            try:
                if getattr(self.cfg.task, "pipette_require_pick_orientation", False):
                    joints = self.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
                    _, matrix = self.rmp_controller.rmp_flow.get_end_effector_pose(joints)
                    target_rotation = R.from_quat(self._pick_orientation()[[1, 2, 3, 0]])
                    angular_error = (target_rotation * R.from_matrix(matrix).inv()).magnitude()
                    self.pick_controller.arrival_orientation_ok = angular_error <= np.radians(2)
                return self.pick_controller.forward(
                    picking_position=pick_position,
                    current_joint_positions=state["joint_positions"],
                    object_size=state["object_size"],
                    object_name=state["object_name"],
                    gripper_control=self.gripper_control,
                    gripper_position=state["gripper_position"],
                    end_effector_orientation=self._pick_orientation(),
                    pre_offset_x=0.0,
                    pre_offset_z=float(getattr(self.cfg.task, "pipette_pregrasp_clearance", 0.02)),
                    after_offset_z=float(getattr(self.cfg.task, "pipette_lift_clearance", 0.30)),
                    gripper_distances=self._gripper_distance,
                )
            except ValueError as exc:
                self.pick_controller._phase_failure_reason = str(exc)
                self.pick_controller._is_done = True
                action = self.pick_controller._null_action(len(state["joint_positions"]))
                return action, self.pick_controller._build_record_array(action, state["joint_positions"])
        transit = getattr(self.cfg.task, "pipette_transit_position", None)
        if transit is not None and self._transit_phase < 2:
            self._transit_steps += 1
            orientation = self._transit_orientation if self._transit_phase == 0 else self._placement_orientation
            try:
                action = self.place_controller._cspace_controller.forward(
                    target_end_effector_position=np.asarray(transit), target_end_effector_orientation=orientation
                )
            except ValueError as error:
                return self._placement_failure(state, str(error))
            joints = self.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
            _, matrix = self.rmp_controller.rmp_flow.get_end_effector_pose(joints)
            error = R.from_quat(orientation[[1, 2, 3, 0]]) * R.from_matrix(matrix).inv()
            # This is an obstacle-free carry waypoint, not the seating gate.
            # Millimetre/one-degree insertion alignment is still enforced later.
            reached = np.linalg.norm(state["gripper_position"] - transit) < 0.010 and error.magnitude() < np.radians(3)
            self._transit_stable = self._transit_stable + 1 if reached else 0
            if self._transit_stable >= 10:
                self._transit_phase += 1
                self._transit_steps = self._transit_stable = 0
            elif self._transit_steps > 1200:
                return self._placement_failure(
                    state,
                    f"Pipette safe transit {self._transit_phase} did not converge: "
                    f"delta={state['gripper_position'] - transit}, angle={np.degrees(error.magnitude()):.2f} deg",
                )
            return action, self.place_controller._build_record_array(
                action, state["joint_positions"], gripper_state=self.place_controller._current_gripper_state
            )
        # Re-measure after rotating above the slot. An AABB centre is not a
        # rigid feature: its offset changes as this asymmetric tool rotates.
        if (
            "object_reference_position" in state
            and self.place_controller._event < 3
            and np.linalg.norm(state["gripper_position"] - state["object_grasp_position"]) > 0.065
        ):
            return self._placement_failure(state, "Pipette grasp lost before release")
        if "object_reference_position" in state and self.place_controller._event == 2:
            error = R.from_quat(state["target_quaternion"]) * R.from_quat(state["object_quaternion"]).inv()
            if error.magnitude() > np.radians(8):
                return self._placement_failure(state, "Pipette tipped against the rack before release")
        front_insertion = bool(getattr(self.cfg.task, "pipette_front_insertion", False))
        alignment_due = (
            self.place_controller._event == 0 and self.place_controller._approach_complete
            if front_insertion
            else self.place_controller._event == 1
        )
        alignment_goal = self._command_seat_position(state)
        if front_insertion:
            alignment_goal += self.place_controller.approach_offset
        approach_height = self._release_clearance() if front_insertion else 0.22
        if alignment_due and not self._placement_offset_refined:
            self._grasp_offset = np.array(state["gripper_position"]) - state.get(
                "object_reference_position", state["object_position"]
            )
            self._placement_offset_refined = True
        if "object_reference_position" in state and alignment_due and not self._alignment_done:
            self._alignment_steps += 1
            correction = R.from_quat(state["target_quaternion"]) * R.from_quat(state["object_quaternion"]).inv()
            if self._alignment_steps == 1:
                joints = self.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
                fk_position, matrix = self.rmp_controller.rmp_flow.get_end_effector_pose(joints)
                self.place_controller._cspace_controller.tcp_local_offset = (
                    R.from_matrix(matrix).inv().apply(state["gripper_position"] - fk_position)
                )
                self._placement_orientation = (correction * R.from_matrix(matrix)).as_quat()[[3, 0, 1, 2]]
                self._grasp_offset = correction.apply(state["gripper_position"] - state["object_reference_position"])
                logger.debug(
                    "Pipette hover alignment: angle_deg={}, delta={}, TCP_local={}",
                    np.degrees(correction.magnitude()),
                    state["object_reference_position"] - self._goal_position(state),
                    self.place_controller._cspace_controller.tcp_local_offset,
                )
            motion = self.place_controller._cspace_controller
            motion.lateral_error = (alignment_goal - state["object_reference_position"])[:2]
            motion.angular_error = correction.as_rotvec()
            if self._alignment_steps % 120 == 0:
                logger.debug(
                    "Pipette alignment servo: angle_deg={}, delta={}",
                    np.degrees(correction.magnitude()),
                    state["object_reference_position"] - self._goal_position(state),
                )
            if (
                self._alignment_steps > 120
                and correction.magnitude() < np.radians(1)
                and np.linalg.norm((state["object_reference_position"] - alignment_goal)[:2]) < 0.001
            ):
                self._alignment_done = True
                joints = self.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
                _, matrix = self.rmp_controller.rmp_flow.get_end_effector_pose(joints)
                self._placement_orientation = (correction * R.from_matrix(matrix)).as_quat()[[3, 0, 1, 2]]
                self._grasp_offset = correction.apply(state["gripper_position"] - state["object_reference_position"])
                motion.lateral_integral.fill(0)
                motion.angular_integral.fill(0)
            elif self._alignment_steps > 480:
                return self._placement_failure(state, "Pipette did not align upright above the slot")
            else:
                try:
                    action = self.place_controller._cspace_controller.forward(
                        target_end_effector_position=alignment_goal
                        + self._grasp_offset
                        + np.array([0, 0, approach_height]),
                        target_end_effector_orientation=self._placement_orientation,
                    )
                except ValueError as exc:
                    return self._placement_failure(state, str(exc))
                return action, self.place_controller._build_record_array(action, state["joint_positions"])
        self._diagnostic_steps = getattr(self, "_diagnostic_steps", 0) + 1
        if "object_reference_position" in state and self._alignment_done:
            try:
                lateral_error, angular_error = self._placement_feedback(state)
            except ValueError as exc:
                return self._placement_failure(state, str(exc))
            self.place_controller._cspace_controller.lateral_error = lateral_error
            self.place_controller._cspace_controller.angular_error = angular_error
            correction = R.from_quat(state["target_quaternion"]) * R.from_quat(state["object_quaternion"]).inv()
            self.place_controller.placement_feature_position = state["object_reference_position"]
            self.place_controller.placement_feature_target = self._command_seat_position(state) + np.array(
                [0, 0, self._release_clearance()]
            )
            self.place_controller.placement_feature_aligned = correction.magnitude() < np.radians(3)
        if self._diagnostic_steps % 100 == 1:
            logger.debug(
                "Pipette place event={}: TCP={}, object={}, axis={}, offset={}, joints={}, object_quat={}",
                self.place_controller._event,
                state["gripper_position"],
                state["object_position"],
                state.get("object_axis"),
                self._grasp_offset,
                state["joint_positions"],
                state.get("object_quaternion"),
            )
        try:
            if getattr(self.cfg.task, "pipette_axial_retreat", False):
                # Pull away along the fingers' approach axis, rather than
                # sliding upwards past the plunger and neighbouring tools.
                self.place_controller.retreat_offset = R.from_quat(self._placement_orientation[[1, 2, 3, 0]]).apply(
                    np.array([0.0, 0.0, -0.12])
                )
            return self.place_controller.forward(
                place_position=self._command_seat_position(state) + self._grasp_offset,
                current_joint_positions=state["joint_positions"],
                gripper_control=self.gripper_control,
                end_effector_orientation=self._placement_orientation,
                gripper_position=state["gripper_position"],
                pre_place_z=(
                    self._release_clearance()
                    if getattr(self.cfg.task, "pipette_front_insertion", False)
                    else 0.22
                    if getattr(self.cfg.task, "reference_geometry", False)
                    else 0.10
                ),
                place_offset_z=self._release_clearance()
                if getattr(self.cfg.task, "reference_geometry", False)
                else 0.055,
            )
        except ValueError as exc:
            return self._placement_failure(state, str(exc))

    def _placement_failure(self, state, reason):
        self.place_controller._phase_failure_reason = str(reason)
        self.place_controller._event = len(self.place_controller._events_dt)
        action = self.place_controller._null_action(len(state["joint_positions"]))
        return action, self.place_controller._build_record_array(action, state["joint_positions"])

    def _trace_grasp(self, state):
        """Measure object motion relative to the hand, not world rotation."""
        if "object_quaternion" not in state:
            return
        if self.current_phase == Phase.PICKING and self.pick_controller._event < 5:
            closed_contact = (
                self.pick_controller._event == 4
                and getattr(self.cfg.task, "pipette_stable_close", False)
                and getattr(self.pick_controller, "_jaw_contact_steps", 0) >= 10
            )
            if not closed_contact:
                return
        if self.current_phase == Phase.PLACING and self.place_controller._event >= 3:
            if self._grasp_reference is not None and not getattr(self, "_grasp_reported", False):
                logger.info(
                    "Pipette grasp summary: object={}, peak_relative_rotation_deg={:.4f}, "
                    "peak_postclose_rotation_deg={:.4f}, sampled_steps={}, "
                    "physical_postclose_peak_deg={}, postclose_phase_peaks={}",
                    state.get("object_path", state.get("object_name", "pipette")),
                    self._grasp_peak_rotation_deg,
                    self._postclose_peak_rotation_deg,
                    self._grasp_trace_steps,
                    getattr(self, "_physical_postclose_peak_deg", None),
                    self._grasp_phase_peaks,
                )
                self._grasp_reported = True
            return  # Motion after opening is not in-hand slip.
        joints = self.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
        _, matrix = self.rmp_controller.rmp_flow.get_end_effector_pose(joints)
        relative = R.from_matrix(matrix).inv() * R.from_quat(state["object_quaternion"])
        if self._grasp_reference is None:
            self._grasp_reference = relative
            self._grasp_peak_rotation_deg = 0.0
            self._postclose_grasp_reference = None
            self._postclose_peak_rotation_deg = 0.0
            self._grasp_phase_peaks = {}
            self._physical_postclose_reference = None
            self._physical_postclose_peak_deg = None
            self._grasp_reported = False
            if os.environ.get("LABUTOPIA_PIPETTE_GRASP_DEBUG") == "1":
                stiffness, damping = self.robot._articulation_view.get_gains()
                logger.info(
                    "Pipette grip drives: stiffness={}, damping={}, mode={}",
                    stiffness,
                    damping,
                    getattr(self.robot, "_gripper_control_mode", "position"),
                )
        self._grasp_trace_steps += 1
        drift = float(np.degrees((relative * self._grasp_reference.inv()).magnitude()))
        self._grasp_peak_rotation_deg = max(self._grasp_peak_rotation_deg, drift)
        # Keep the original contact-onset metric and its failure guard intact.
        # Separately measure drift after the full object/TCP stable-close gate,
        # immediately before the first lift command is issued.
        if self.current_phase == Phase.PLACING or self.pick_controller._event >= 5:
            if self._postclose_grasp_reference is None:
                self._postclose_grasp_reference = relative
            postclose_drift = float(np.degrees((relative * self._postclose_grasp_reference.inv()).magnitude()))
            self._postclose_peak_rotation_deg = max(self._postclose_peak_rotation_deg, postclose_drift)
            phase_key = "picking" if self.current_phase == Phase.PICKING else f"placing_{self.place_controller._event}"
            self._grasp_phase_peaks[phase_key] = max(self._grasp_phase_peaks.get(phase_key, 0), postclose_drift)
            if "physical_hand_quaternion" in state:
                physical_relative = R.from_quat(state["physical_hand_quaternion"]).inv() * R.from_quat(
                    state["object_quaternion"]
                )
                if self._physical_postclose_reference is None:
                    self._physical_postclose_reference = physical_relative
                    self._physical_postclose_peak_deg = 0.0
                physical_drift = float(
                    np.degrees((physical_relative * self._physical_postclose_reference.inv()).magnitude())
                )
                self._physical_postclose_peak_deg = max(self._physical_postclose_peak_deg, physical_drift)
        if (self._grasp_trace_steps % 30 == 1 or drift > 6) and os.environ.get("LABUTOPIA_PIPETTE_GRASP_DEBUG") == "1":
            logger.info(
                "Pipette relative grip rotation: phase={}, degrees={:.2f}, fingers={}, body_rotvec_deg={}",
                self.current_phase.value,
                drift,
                state["joint_positions"][-2:],
                np.degrees((self._grasp_reference.inv() * relative).as_rotvec()),
            )
            if self.current_phase == Phase.PICKING and isinstance(
                self.pick_controller._cspace_controller, PipettePoseMotion
            ):
                motion = self.pick_controller._cspace_controller
                fk, _ = self.rmp_controller.rmp_flow.get_end_effector_pose(joints)
                logger.info(
                    "Pipette lift tracking: FK={}, TCP={}, local={}, seed_error={}, bias={}, joints={}",
                    fk,
                    state["gripper_position"],
                    motion.tcp_local_offset,
                    motion.seed - joints,
                    motion.tracking_bias,
                    joints,
                )
        return drift
