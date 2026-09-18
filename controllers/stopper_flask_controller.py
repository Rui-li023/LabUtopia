"""Place a stopper at the flask mouth with the shared precision placement flow."""

import random

import numpy as np
from scipy.spatial.transform import Rotation as R

from .flask_to_cork_controller import FlaskToCorkTaskController, Phase


class StopperFlaskTaskController(FlaskToCorkTaskController):
    """Top-down stopper insertion followed by release and a stable-seat check."""

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        self.pick_controller.grasp_position_threshold = 0.006
        self.pick_controller.arrival_stable_steps = 3
        if bool(getattr(cfg.task, "physical_insertion", False)):
            self.pick_controller.pick_z_offset_override = 0.0
            self.pick_controller.hold_pose_during_close = True
            self.pick_controller.grasp_position_threshold = 0.003
            self.pick_controller.arrival_stable_steps = 5
            self.pick_controller.close_contact_gap_range = (0.030, 0.038)
            self.pick_controller.close_stable_steps = 15
            self.place_controller.precision_position_threshold = 0.002
            self.place_controller.precision_height_threshold = 0.0015

    def _goal_position(self, state):
        # Both positions are geometry centres, not USD origins. The stopper's
        # lower tapered section enters the mouth by the configured depth.
        goal = np.array(state["target_position"], copy=True)
        depth = float(getattr(self.cfg.task, "insertion_depth", 0.008))
        if state.get("target_mouth_position") is not None:
            axis = np.asarray(state["target_up"], dtype=float)
            return np.asarray(state["target_mouth_position"]) + axis * (0.5 * self.initial_size[2] - depth)
        goal[2] += 0.5 * state["target_size"][2] + 0.5 * self.initial_size[2] - depth
        return goal

    def _check_placed_on_target(self) -> bool:
        if not bool(getattr(self.cfg.task, "physical_insertion", False)):
            return super()._check_placed_on_target()
        state = self.state
        tip, mouth = state.get("object_tip_position"), state.get("target_mouth_position")
        if tip is None or mouth is None:
            self._last_failure_reason = "Insertion requires the physical stopper Tip and flask Mouth markers"
            return False
        tip, mouth = np.asarray(tip), np.asarray(mouth)
        axis = np.asarray(state["target_up"])
        up = np.asarray(state["object_up"])
        delta = tip - mouth
        depth = -float(np.dot(delta, axis))
        lateral = float(np.linalg.norm(delta + depth * axis))
        angle = float(np.degrees(np.arccos(np.clip(np.dot(up, axis), -1, 1))))
        previous = self._previous_object_position
        motion = float(np.linalg.norm(tip - previous)) if previous is not None else float("inf")
        self._previous_object_position = tip.copy()
        released = float(np.linalg.norm(state["gripper_position"] - state["object_position"])) > 0.07
        minimum = float(getattr(self.cfg.task, "minimum_insertion_depth", 0.012))
        maximum = float(getattr(self.cfg.task, "maximum_insertion_depth", 0.022))
        passed = bool(
            minimum <= depth <= maximum
            and lateral < float(getattr(self.cfg.task, "placement_xy_tolerance", 0.003))
            and angle < float(getattr(self.cfg.task, "placement_tilt_tolerance_deg", 8))
            and axis[2] > np.cos(np.radians(10))
            and released
            and motion < 0.0005
        )
        self._last_failure_reason = (
            ""
            if passed
            else (
                f"insertion: depth={depth:.4f}, lateral={lateral:.4f}, angle={angle:.2f}, "
                f"released={released}, motion={motion:.5f}"
            )
        )
        return passed

    def _capture_grasp_offset(self, state):
        offset = super()._capture_grasp_offset(state)
        if state.get("object_tip_position") is not None:
            self._grasp_tip_distance = float(np.linalg.norm(state["gripper_position"] - state["object_tip_position"]))
            self._insertion_alignment_steps = 0
            self._insertion_approach_ready = False
            joints = self.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
            hand_position, hand_matrix = self.rmp_controller.rmp_flow.get_end_effector_pose(joints)
            self._transfer_waypoint = np.asarray(hand_position).copy()
            self._insertion_hand_target = self._transfer_waypoint.copy()
            self._placement_hand_orientation = R.from_matrix(hand_matrix).as_quat()[[3, 0, 1, 2]]
            self._transfer_step = 0.0
        return offset

    def _insertion_pose(self, state):
        """Correct the held stopper's measured tip/axis, then freeze after release."""
        place = self.place_controller
        if place._event <= 1:
            tip = np.asarray(state["object_tip_position"])
            axis = np.asarray(state["target_up"])
            target = np.asarray(state["target_mouth_position"]) - axis * float(self.cfg.task.insertion_depth)
            offset = np.asarray(state["gripper_position"]) - tip
            if np.linalg.norm(offset) > self._grasp_tip_distance + 0.02:
                place._phase_failure_reason = "Stopper left the gripper before insertion release"
                place._event = len(place._events_dt)
                return self._insertion_hand_target, self._placement_hand_orientation
            alignment, _ = R.align_vectors([axis], [state["object_up"]])
            angle = alignment.magnitude()
            rotation = R.from_rotvec(alignment.as_rotvec() * min(1.0, np.radians(3) / max(angle, 1e-9)))
            # RMPFlow's tool frame is not the USD panda_hand frame. Apply the
            # measured object-axis correction in the solver's own FK frame.
            joints = self.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
            hand_position, hand_matrix = self.rmp_controller.rmp_flow.get_end_effector_pose(joints)
            if place._event == 1 and not self._insertion_approach_ready:
                delta = tip - np.asarray(state["target_mouth_position"])
                lateral = np.linalg.norm(delta - np.dot(delta, axis) * axis)
                aligned = lateral < 0.002 and angle < np.radians(3)
                self._insertion_alignment_steps = self._insertion_alignment_steps + 1 if aligned else 0
                self._insertion_approach_ready = self._insertion_alignment_steps >= 5
                if not self._insertion_approach_ready:
                    # Align above the lip before descending, so a tilted cork
                    # cannot lever the freely resting flask out of its ring.
                    target = target + axis * 0.06
            self._insertion_hand_target = target + rotation.apply(np.asarray(hand_position) - tip)
            self._placement_hand_orientation = (rotation * R.from_matrix(hand_matrix)).as_quat()[[3, 0, 1, 2]]
            place.placement_feature_position = tip
            place.placement_feature_target = np.asarray(state["target_mouth_position"]) - axis * float(
                self.cfg.task.insertion_depth
            )
            place.placement_feature_aligned = self._insertion_approach_ready and angle < np.radians(5)
        return self._insertion_hand_target, self._placement_hand_orientation

    def _sample_phase_instruction(self, phase):
        if phase == Phase.PICKING:
            templates = self._build_instruction_templates(
                "Pick up the rubber stopper from the table",
                "Pick up the rubber stopper from the table and lift it",
            )
        else:
            templates = self._build_instruction_templates(
                "Place the rubber stopper into the flask opening",
                "Insert the rubber stopper into the mouth of the round-bottom flask",
            )
        return random.choice(templates)

    def _forward_active(self, state):
        orientation = R.from_euler("xyz", np.radians([0, 180, 0])).as_quat()
        physical = bool(getattr(self.cfg.task, "physical_insertion", False))
        measured_hand = state["gripper_position"]
        picking_position = state["object_position"]
        if physical:
            joints = self.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
            measured_hand, _ = self.rmp_controller.rmp_flow.get_end_effector_pose(joints)
            measured_hand = np.asarray(measured_hand)
            # Convert the physical grasp marker to the controlled tool frame;
            # arrival checks must use that same frame, not panda_hand/tool_center.
            picking_position = state["object_grasp_position"] + measured_hand - state["gripper_position"]
        if self.current_phase == Phase.PICKING:
            return self.pick_controller.forward(
                picking_position=picking_position,
                current_joint_positions=state["joint_positions"],
                object_size=state["object_size"],
                object_name=state["object_name"],
                gripper_control=self.gripper_control,
                gripper_position=measured_hand,
                end_effector_orientation=orientation,
                pre_offset_x=0.0,
                pre_offset_z=0.03,
                after_offset_z=0.14 if physical else 0.25,
                gripper_distances=self._gripper_distance,
            )
        position = self._goal_position(state) + self._grasp_offset
        if physical:
            position, orientation = self._insertion_pose(state)
            if self.place_controller._event == 0:
                position = self._transport_position(position)
        return self.place_controller.forward(
            place_position=position,
            current_joint_positions=state["joint_positions"],
            gripper_control=self.gripper_control,
            end_effector_orientation=orientation,
            gripper_position=measured_hand,
            pre_place_z=0.06 if physical else 0.15,
            place_offset_z=0.0 if physical else 0.002,
        )

    def _transport_position(self, insertion_position):
        """Ramp the commanded hover path to avoid throwing a light tapered cork."""
        hover = np.asarray(insertion_position) + np.array([0, 0, 0.06])
        delta = hover - self._transfer_waypoint
        distance = float(np.linalg.norm(delta))
        self._transfer_step = min(self._transfer_step + 0.0001, 0.002)
        self._transfer_waypoint += delta * min(1.0, self._transfer_step / max(distance, 1e-9))
        self.place_controller.approach_arrival_allowed = np.linalg.norm(hover - self._transfer_waypoint) < 0.001
        # PlaceController adds pre_place_z itself.
        return self._transfer_waypoint - np.array([0, 0, 0.06])
