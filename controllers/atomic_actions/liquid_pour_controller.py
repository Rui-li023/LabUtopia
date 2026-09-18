"""Measured-rim pouring: rotate the grasp around a lip above the receiver.

Only joint targets are commanded. Liquid observations govern the drain/return
transition; particles and container poses are never manipulated here.
"""

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from .atomic_base_controller import AtomicBaseController


def configure_liquid_pick(pick):
    """Use physical arrival and jaw contact before lifting a water-filled beaker."""
    pick.adaptive_phase_completion = True
    pick.arrival_stable_steps = 3
    pick.lock_final_approach_target = True
    pick.hold_pose_during_close = True
    pick.require_pregrasp_xyz = True
    pick.grasp_position_threshold = 0.005
    pick.close_contact_gap_range = (0.06, 0.079)
    pick.close_stable_steps = 15
    pick._events_dt = [0.001, 0.001, 0.001, 0.02, 0.01, 0.001, 0.02]


def forward_liquid_pick(pick, state, gripper_control, orientation, grip_distance, grasp_depth=0.045):
    motion = pick._cspace_controller
    joints = motion.articulation_rmp.get_active_joints_subset().get_joint_positions()
    hand, _ = motion.rmp_flow.get_end_effector_pose(joints)
    # Place the actual finger center at mid-wall, expressed in the RMP tool frame.
    liquid = state["liquid"]
    grasp = np.asarray(liquid["source_mouth"]) - np.asarray(liquid["source_up"]) * grasp_depth
    position = grasp + np.asarray(hand) - np.asarray(state["gripper_position"])
    return pick.forward(
        picking_position=position,
        current_joint_positions=state["joint_positions"],
        object_size=state["object_size"],
        object_name=state["object_name"],
        gripper_control=gripper_control,
        gripper_position=np.asarray(hand),
        end_effector_orientation=orientation,
        after_offset_z=0.18,
        pre_offset_x=0.07,
        pre_offset_z=0.03,
        gripper_distances=grip_distance,
    )


class LiquidPourController(AtomicBaseController):
    def __init__(self, name, cspace_controller, cfg):
        super().__init__(name, cspace_controller, default_events_dt=[0.001] * 5)
        self.cfg = cfg
        self.reset()

    def reset(self, events_dt=None):
        super().reset(events_dt)
        self.angle = 0.0
        self.frames = 0
        self.stable = 0
        self.direction = None
        self.radial_local = None
        self.waypoint = None
        self._is_done = False
        self._phase_failure_reason = ""
        self._initial_grasp_distance = None
        self._grasp_orientation = None
        self._return_clearance_ready = False
        self._return_clearance_stable = 0
        self._return_retreat_direction = None

    def forward(self, state):
        liquid = state["liquid"]
        joints = self._cspace_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
        hand, hand_matrix = self._cspace_controller.rmp_flow.get_end_effector_pose(joints)
        hand = np.asarray(hand)
        up = np.asarray(liquid["source_up"])
        mouth = np.asarray(liquid["source_mouth"])
        basis = np.asarray(liquid["source_basis"])
        options = self.cfg.liquid
        grasp_distance = float(np.linalg.norm(hand - mouth))
        if self._initial_grasp_distance is None:
            self._initial_grasp_distance = grasp_distance
        slip_limit = getattr(options, "max_grasp_separation_change", None)
        target_tilt = getattr(options, "max_target_tilt_deg", None)
        if target_tilt is not None and liquid["target_up"][2] < np.cos(np.deg2rad(float(target_tilt))):
            self._phase_failure_reason = "Receiver tipped during liquid transport/pour"
        elif slip_limit is not None and abs(grasp_distance - self._initial_grasp_distance) > float(slip_limit):
            self._phase_failure_reason = "Bottle slipped from the gripper during liquid transport/pour"
        if self._phase_failure_reason:
            self._is_done = True
            action = self._null_action(len(state["joint_positions"]))
            return action, self._build_record_array(action, state["joint_positions"])
        if self.direction is None:
            # Roll around the current horizontal approach axis, using the side
            # with greater joint-7 headroom (Franka wrist limits are +/-2.8973).
            axis = np.asarray(hand_matrix)[:, 2].copy()
            axis[2] = 0
            axis /= np.linalg.norm(axis)
            if float(joints[6]) > 0:
                axis *= -1
            self.direction = np.cross(axis, [0, 0, 1])
            self._grasp_orientation = Rotation.from_matrix(hand_matrix)
            self.radial_local = self.direction @ basis.T
            self.waypoint = mouth + float(liquid["source_radius"]) * self.direction
            logger.info("[liquid-pour] wrist={:.2f} roll_direction={}", joints[6], self.direction)
        radial = self.radial_local @ basis
        if bool(getattr(options, "axisymmetric_source", False)):
            # An unspouted conical flask has no privileged azimuth. Derive its
            # pouring lip from the tilt plane, not from incidental axial spin.
            radial = np.cross(np.cross([0.0, 0.0, 1.0], self.direction), up)
        radial /= np.linalg.norm(radial)
        lip = mouth + float(liquid["source_radius"]) * radial
        # Keep the upright cup's bottom above the receiver rim, then lower its
        # pouring lip as it tilts. A fixed 12 cm drop caused avoidable splashes.
        lip_height = float(getattr(options, "pour_lip_height", 0.07)) + float(
            getattr(options, "pour_upright_clearance", 0.04)
        ) * max(np.cos(self.angle), 0.0)
        return_lift = float(getattr(options, "pour_return_lift", 0.0))
        if self._event >= 3:
            lip_height += return_lift
        target = np.asarray(liquid["target_mouth"]) + np.array([0.0, 0.0, lip_height])
        retreat = float(getattr(options, "pour_return_retreat", 0.0))
        if self._event >= 3 and retreat > 0:
            if self._return_retreat_direction is None:
                away = np.asarray(self._cspace_controller._default_position) - np.asarray(liquid["target_mouth"])
                away[2] = 0.0
                self._return_retreat_direction = away / max(np.linalg.norm(away), 1e-9)
            target += self._return_retreat_direction * retreat
        error = np.linalg.norm(lip - target)
        self.frames += 1
        if self._event == 0:
            if self.frames > int(getattr(options, "pour_approach_timeout", 10000)):
                self._phase_failure_reason = f"Pour approach did not converge: lip error={error:.4f} m"
                self._is_done = True
            delta = target - self.waypoint
            self.waypoint += delta * min(1.0, 0.0015 / max(np.linalg.norm(delta), 1e-9))
            aligned = error < float(getattr(options, "pour_approach_tolerance", 0.008)) and up[2] > 0.99
            self.stable = self.stable + 1 if aligned else 0
            if self.stable >= 30:
                self._event, self.frames = 1, 0
        elif self._event == 1:
            maximum_angle = np.deg2rad(float(getattr(options, "pour_max_angle_deg", 115)))
            self.angle = min(maximum_angle, self.angle + 0.0035)
            if self.angle >= maximum_angle:
                self._event, self.frames = 2, 0
        elif self._event == 2:
            emptied = liquid["source_fraction"] < float(getattr(options, "pour_empty_threshold", 0.08))
            # Preserve an explicit drain dwell in the recorded policy. Small
            # contact differences in open-loop replay must not cause departure
            # while the real flask is still emptying above the receiver.
            minimum_dwell = int(getattr(options, "pour_min_drain_steps", 0))
            if (emptied and self.frames >= minimum_dwell) or self.frames >= int(getattr(options, "pour_drain_steps", 360)):
                self._event, self.frames = 3, 0
        elif self._event == 3:
            # Raise the emptied flask clear of the receiver BEFORE its broad
            # bottom swings down. Measured arrival, not a fixed waiting timer.
            if return_lift > 0 and not self._return_clearance_ready:
                self._return_clearance_stable = self._return_clearance_stable + 1 if error < 0.015 else 0
                self._return_clearance_ready = self._return_clearance_stable >= 30
                if self.frames > 600:
                    self._phase_failure_reason = "Return-clearance lift did not converge"
                    self._is_done = True
            else:
                self.angle = max(0.0, self.angle - 0.004)
                if self.angle <= 0.0:
                    self._event, self.frames = 4, 0
        elif self._event == 4 and self.frames >= 240:
            self._is_done = True
        if self._event > 0:
            if self._event == 3 and return_lift > 0 and not self._return_clearance_ready:
                delta = target - self.waypoint
                self.waypoint += delta * min(1.0, 0.0015 / max(np.linalg.norm(delta), 1e-9))
            else:
                self.waypoint = target.copy()
        # Hold the commanded upright pose until the shared stable-liquid gate
        # succeeds or the task's finite timeout reports the physical failure.
        desired_up = np.cos(self.angle) * np.array([0, 0, 1]) + np.sin(self.angle) * self.direction
        desired_radial = np.cos(self.angle) * self.direction - np.sin(self.angle) * np.array([0, 0, 1])
        correction, _ = Rotation.align_vectors([desired_up, desired_radial], [up, radial])
        if bool(getattr(options, "fixed_grasp_orientation", False)):
            # An absolute wrist-roll trajectory avoids accumulating axial
            # bottle spin into unrelated arm joints near the pour singularity.
            roll_axis = np.cross([0.0, 0.0, 1.0], self.direction)
            desired_hand = Rotation.from_rotvec(roll_axis * self.angle) * self._grasp_orientation
            correction = desired_hand * Rotation.from_matrix(hand_matrix).inv()
        # The desired angle already ramps slowly. Clamping its error relative
        # to the measured hand makes RMPFlow chase a tiny moving target and can
        # leave the real cup tens of degrees behind the commanded pour angle.
        position = self.waypoint + correction.apply(hand - lip)
        orientation = (correction * Rotation.from_matrix(hand_matrix)).as_quat()[[3, 0, 1, 2]]
        action = self._cspace_controller.forward(
            target_end_effector_position=position,
            target_end_effector_orientation=orientation,
        )
        if self.frames % 120 == 0:
            logger.info(
                "[liquid-pour] phase={} angle={:.1f} lip_error={:.3f} up_z={:.3f} mouth={} target={}",
                self._event,
                np.degrees(self.angle),
                error,
                up[2],
                np.round(mouth, 4),
                np.round(target, 4),
            )
        return action, self._build_record_array(action, state["joint_positions"])
