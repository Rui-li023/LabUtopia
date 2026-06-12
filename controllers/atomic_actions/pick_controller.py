from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing

from .atomic_base_controller import AtomicBaseController
from robots.base_robot import BaseRobot, GRIPPER_CLOSED, GRIPPER_OPEN


class PickController(AtomicBaseController):
    """State machine for picking objects (7 phases).

    Phase 0: Move above object.  Phase 1: Lower to pre-grasp.
    Phase 2: Position for grasp.  Phase 3: Wait for settle.
    Phase 4: Close gripper.  Phase 5: Lift.  Phase 6: Done.

    Per-episode randomization:
      - pre/after Z offset noise (±0.04 m)
      - end-effector orientation perturbation (±15 deg)
    """

    DEFAULT_DT = [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.006]

    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.01,
        robot: typing.Optional[BaseRobot] = None,
    ) -> None:
        super().__init__(
            name=name,
            cspace_controller=cspace_controller,
            events_dt=events_dt,
            default_events_dt=self.DEFAULT_DT,
            robot=robot,
            position_threshold=position_threshold,
            require_gripper=True,
        )
        self.object_size = None
        self._robot_position = None

        # Per-episode noise (sampled lazily)
        self._pre_offset_z_noise = 0.0
        self._after_offset_z_noise = 0.0
        self._orientation_angle_deg = 0.0

    def set_robot_position(self, position) -> None:
        """Set the robot base position used to compute relative pick geometry.

        Used by mobile manipulation, where the base moves between steps, so the
        atomic pick must re-reference the (moving) base on each call.
        """
        self._robot_position = position

    # ── Randomization ────────────────────────────────────────────

    def _sample_randomization(self):
        # Positive-only lift noise: never reduce the lift height below the default,
        # since downstream phases (e.g. pour) require >= 0.12 m clearance.
        self._pre_offset_z_noise = self._uniform(0.0, 0.04)
        self._after_offset_z_noise = self._uniform(0.0, 0.04)
        # Yaw noise around the WORLD vertical: varies grasp heading while
        # keeping the held object level (tool-Z rotation tilts side grasps).
        self._orientation_angle_deg = self._noisy(0.0, 15.0)

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self,
        picking_position: np.ndarray,
        current_joint_positions: np.ndarray,
        object_name: str,
        object_size: np.ndarray,
        gripper_control,
        gripper_position: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        pre_offset_z: float = 0.12,
        after_offset_z: float = 0.15,
        pre_offset_x: float = 0.1,
        gripper_distances: float = None,
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        self.object_size = object_size
        n = current_joint_positions.shape[0]

        # Guard: if the task could not resolve the target object's pose (e.g.
        # a USD/task mismatch returns None), bail out with a null action
        # instead of crashing Isaac Sim with a TypeError deep in the C++ stack.
        if picking_position is None or object_size is None:
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions,
                gripper_state=self._current_gripper_state)

        # First call: open gripper
        if self._start:
            self._start = False
            self._open_gripper()
            action = self._null_action(n)
            return action, self._build_record_array(action, current_joint_positions,
                                                    gripper_state=GRIPPER_OPEN)

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        self._ensure_randomization()

        # Apply per-episode noise
        pre_offset_z = max(0.0, pre_offset_z + self._pre_offset_z_noise)
        after_offset_z = max(0.0, after_offset_z + self._after_offset_z_noise)
        end_effector_orientation = self._apply_world_yaw(
            end_effector_orientation, self._orientation_angle_deg)

        action = self._execute_phase(
            picking_position, end_effector_orientation,
            current_joint_positions, object_name, gripper_control,
            gripper_position, pre_offset_z, after_offset_z, pre_offset_x,
            gripper_distances)

        self._advance_state()
        return action, self._build_record_array(
            action, current_joint_positions,
            gripper_state=self._current_gripper_state)

    # ── Phase execution ──────────────────────────────────────────

    def _calculate_approach_direction(self, picking_position):
        if self._robot_position is None:
            return np.array([-1, 0, 0])
        h = picking_position[:2] - self._robot_position[:2]
        n = np.linalg.norm(h)
        return -h / n if n > 0 else np.array([-1, 0, 0])

    def _execute_phase(self, pos, orient, jpos, obj_name, grip_ctrl,
                       grip_pos, pre_z, after_z, pre_x, grip_dist):
        approach = self._calculate_approach_direction(pos)
        n = jpos.shape[0]
        su = get_stage_units()

        if self._event == 0:
            target = pos + approach * (pre_x / su)
            target[2] += self.object_size[2] + pre_z
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient)
            if self._xy_reached(grip_pos, target):
                self._next_event()
            return action

        elif self._event == 1:
            target = pos + approach * (pre_x / su)
            target[2] += self.get_pickprez_offset(obj_name) / su
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient)
            if self._xy_reached(grip_pos, target):
                self._next_event()
            return action

        elif self._event == 2:
            pos[2] += self.get_pickz_offset(obj_name) / su
            action = self._cspace_controller.forward(
                target_end_effector_position=pos,
                target_end_effector_orientation=orient)
            if self._xyz_reached(grip_pos, pos):
                self._next_event()
            return action

        elif self._event == 3:
            return self._null_action(n)

        elif self._event == 4:
            # Distance-based close when a grasp width is supplied: stops the
            # fingers on contact instead of slamming to 0 (which ejects light/
            # round objects). Falls back to the binary close otherwise.
            if grip_dist is not None and grip_dist > 0:
                self._robot.close_gripper_to_distance(grip_dist)
                self._current_gripper_state = GRIPPER_CLOSED
                # Record the commanded grip width so replay reproduces this exact
                # (firm) grasp via the normalized gripper channel.
                self._last_gripper_target_m = grip_dist
            else:
                self._close_gripper()
            self._lift_target = pos.copy()
            self._lift_target[2] += after_z / su
            if "glass" in obj_name:
                # Attach the PARENT prim, not the Cylinder mesh: the world-space
                # follow in grapper_manager writes the prim's translate op, which
                # for the parent equals its world origin. The mesh child's
                # translate is in parent-LOCAL units (scaled), so attaching it
                # moves the rod by ~scale× too little and it never reaches the
                # beaker (rod stayed at the pick spot, xy≈0.5 vs <0.04 needed).
                grip_ctrl.add_object_to_gripper(
                    "/World/glass_rod",
                    self._robot.gripper_center_prim_path)
            return self._null_action(n)

        elif self._event == 5:
            action = self._cspace_controller.forward(
                target_end_effector_position=self._lift_target,
                target_end_effector_orientation=orient)
            if self._xyz_reached(grip_pos, self._lift_target):
                self._next_event()
            return action

        else:
            return self._null_action(n)

    # ── Offset lookup tables ─────────────────────────────────────

    def get_gripper_distance(self, item_name):
        table = {
            "rod": 0.003, "tube": 0.01, "beaker": 0.022,
            "beaker_l": 0.03, "beaker_04": 0.025, "beaker_05": 0.025,
            "beaker_03": 0.025, "Erlenmeyer flask": 0.018,
            "pipette": 0.008, "microscope slide": 0.002,
            "graduated_cylinder_01": 0.005, "graduated_cylinder_02": 0.018,
            "graduated_cylinder_04": 0.030,
        }
        return table.get(item_name.lower(), 0.0)

    def get_pickz_offset(self, item_name):
        table = {
            "conical_bottle02": 0.065, "conical_bottle03": 0.07,
            "conical_bottle04": 0.08, "beaker": 0.0, "beaker_04": 0.0,
            "beaker_05": 0.0, "beaker_03": 0.0, "beaker2": 0.0,
            "beaker_2": 0.0, "beaker_l": 0.02,
            "graduated_cylinder_01": 0.0, "graduated_cylinder_02": 0.0,
            "graduated_cylinder_03": 0.0, "graduated_cylinder_04": 0.0,
            "volume_flask": 0.05, "glass_rod": 0.02, "round_bottomflask": 0.03,
            "round_bottom_flask": 0.025,
            "pipette": 0.0,
        }
        item_lower = item_name.lower()
        if item_lower in table:
            return table[item_lower]
        for key, val in table.items():
            if key in item_lower:
                return val
        return self.object_size[2] * 2 / 5

    def get_pickprez_offset(self, item_name):
        table = {
            "volume_flask": 0, "beaker2": 0.05, "round_bottomflask": 0.04,
            "round_bottom_flask": 0.03,
            "conical_bottle03": 0.07, "conical_bottle04": 0.08,
            "graduated_cylinder_01": 0.05, "graduated_cylinder_02": 0.03,
            "graduated_cylinder_03": 0.03, "graduated_cylinder_04": 0.03,
            "pipette": 0.0,
        }
        item_lower = item_name.lower()
        if item_lower in table:
            return table[item_lower]
        for key, val in table.items():
            if key in item_lower:
                return val
        return self.object_size[2] * 2 / 3

    # ── Reset ────────────────────────────────────────────────────

    def reset(self, events_dt=None):
        super().reset(events_dt)
        self.object_size = None
        self._robot_position = None
        self._pre_offset_z_noise = 0.0
        self._after_offset_z_noise = 0.0
        self._orientation_angle_deg = 0.0
