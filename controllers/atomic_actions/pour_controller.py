from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.utils.types import ArticulationAction
import numpy as np
import typing
from scipy.spatial.transform import Rotation as R

from .atomic_base_controller import AtomicBaseController, GRIPPER_MAX_OPEN
from robots.base_robot import GRIPPER_CLOSED

CONTROL_FREQUENCY = 60
PHYSICS_DT = 1.0 / CONTROL_FREQUENCY


class PourController(AtomicBaseController):
    """State machine for pouring liquid (6 phases, velocity-controlled pour).

    Phase 0: Move above target.  Phase 1: Adjust height.
    Phase 2: Pour (positive vel).  Phase 3: Pause.
    Phase 4: Pour reverse (negative vel).  Phase 5: Hold.

    Per-episode randomization:
      - heights (0.3-0.4 m and 0.1-0.2 m)
      - pour speed factor (0.8x-1.2x)
      - x offset noise (±0.01 m)
      - orientation noise (±5 deg per axis)
    """

    DEFAULT_DT = [0.002, 0.01, 0.009, 0.005, 0.009, 0.5]

    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1,
        position_threshold: float = 0.006,
        control_frequency: float = CONTROL_FREQUENCY,
        position_pour: bool = False,
        pour_angle_rad: float = 2.0,
    ) -> None:
        # Opt-in: drive the pour tilt with POSITION control on the wrist (joint 6)
        # instead of VELOCITY. The velocity pour records the *integrated commanded
        # velocity* as the wrist angle, which drifts from the real joint (tracking
        # error + held-beaker inertia), so open-loop replay reproduces a different
        # tilt and the beaker slips. A position pour ramps joint 6 to a fixed angle
        # via self._t (phase progress) and records the real commanded position, so
        # replay reproduces it exactly. Requires current_joint_positions in forward.
        self._position_pour = bool(position_pour)
        self._pour_angle_rad = float(pour_angle_rad)
        self._pour_arm0 = None
        # Articulation DOF index of the wrist (panda_joint7) used for
        # switch_dof_control_mode. 6 on a bare Franka; mobile bases with
        # leading base DOFs must set the true articulation index (e.g. 9 on
        # Ridgebase). Subset-relative indexing of jp/vels (always [6]) is
        # unaffected — callers pass ARM-subset arrays.
        self.wrist_dof_index = 6
        dt = events_dt
        if dt is None:
            dt = [d / speed for d in self.DEFAULT_DT]
        super().__init__(
            name=name,
            cspace_controller=cspace_controller,
            events_dt=dt,
            default_events_dt=self.DEFAULT_DT,
            position_threshold=position_threshold,
        )
        self._physics_dt = 1.0 / control_frequency
        self._pour_default_speed = -120.0 / 180.0 * np.pi

        # Per-episode noise. Keep the first hover reachable for Franka when
        # the target container is already near tabletop height (~0.87m).
        self._height_range_1 = (0.15, 0.25)
        self._height_range_2 = (0.1, 0.2)
        self._random_height_1 = self._uniform(*self._height_range_1)
        self._random_height_2 = self._uniform(*self._height_range_2)
        self._speed_factor = 1.0
        self._x_offset_noise = 0.0
        self._orient_noise = np.zeros(3)  # per-axis deg noise for [x, y, z]

        self._last_arm_positions = None

    # ── Randomization ────────────────────────────────────────────

    def _sample_randomization(self):
        self._random_height_1 = self._uniform(*self._height_range_1)
        self._random_height_2 = self._uniform(*self._height_range_2)
        # Narrower speed range so the return rotation always completes within the state machine.
        self._speed_factor = self._uniform(0.95, 1.1)
        self._x_offset_noise = self._noisy(0.0, 0.01)
        # Only randomize rotation around the world Z axis: tilting around X/Y misaligns
        # the bottle from the target XY and produces large pour-distance errors.
        self._orient_noise = np.array([
            0.0,
            0.0,
            self._noisy(0.0, 5.0),
        ])

    # ── Record array (velocity-aware override) ───────────────────

    def _build_record_array(self, action, current_joint_positions=None,
                            gripper_state=None):
        if gripper_state is None:
            gripper_state = GRIPPER_CLOSED
        self._last_gripper_state = gripper_state

        jp = action.joint_positions
        jv = action.joint_velocities

        if jp is not None and current_joint_positions is not None:
            fallback = (self._last_arm_positions
                        if self._last_arm_positions is not None
                        else current_joint_positions[:7])
            arm = fallback.copy().astype(np.float64)
            for i in range(min(len(jp), 7)):
                if jp[i] is not None:
                    arm[i] = float(jp[i])
            self._last_arm_positions = arm
        elif jp is not None:
            arm = np.array([float(p) if p is not None else 0.0 for p in jp[:7]])
            if self._last_arm_positions is not None:
                full = self._last_arm_positions.copy().astype(np.float64)
                for i in range(min(len(arm), 7)):
                    full[i] = arm[i]
                arm = full
            self._last_arm_positions = arm
        elif jv is not None:
            base = (self._last_arm_positions if self._last_arm_positions is not None
                    else (current_joint_positions[:7]
                          if current_joint_positions is not None else None))
            if base is not None:
                arm = np.asarray(base, dtype=np.float64).copy()
                for i in range(min(len(jv), 7)):
                    if jv[i] is not None:
                        arm[i] += float(jv[i]) * self._physics_dt
                self._last_arm_positions = arm
            elif self._last_arm_positions is not None:
                arm = self._last_arm_positions.copy()
            else:
                return None
        elif self._last_arm_positions is not None:
            arm = self._last_arm_positions.copy()
        elif current_joint_positions is not None:
            arm = current_joint_positions[:7].copy()
            self._last_arm_positions = arm
        else:
            return None

        record = np.zeros(8, dtype=np.float64)
        record[:7] = arm[:7]
        # Gripper channel: mirror AtomicBaseController._build_record_array.
        # The old hard-coded `float(gripper_state)` wrote 1.0 (binary full
        # close) for the entire pour while the hand physically held the pick's
        # contact-stop WIDTH — the record lied about the executed command.
        # Replay then reproduced the lie, position-slamming the fingers to 0
        # mid-pour and ejecting the held container (clean_beaker's beaker flew
        # to the floor; liquid_mixing's first pick never survived its pour).
        if AtomicBaseController.record_commanded_gripper and self._robot is not None:
            opening = float(np.clip(self._robot.get_gripper_commanded_opening(),
                                    0.0, GRIPPER_MAX_OPEN))
            record[7] = float(np.clip(1.0 - opening / GRIPPER_MAX_OPEN, 0.0, 1.0))
        elif current_joint_positions is not None and len(current_joint_positions) > 7:
            opening = float(np.clip(current_joint_positions[7], 0.0, GRIPPER_MAX_OPEN))
            record[7] = float(np.clip(1.0 - opening / GRIPPER_MAX_OPEN, 0.0, 1.0))
        else:
            record[7] = float(gripper_state)
        return record

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self,
        articulation_controller: ArticulationController,
        source_size: np.ndarray,
        target_position: np.ndarray,
        current_joint_velocities: np.ndarray,
        gripper_position: np.ndarray,
        source_name: str = None,
        pour_speed: float = None,
        current_joint_positions: typing.Optional[np.ndarray] = None,
        target_end_effector_orientation=None,
    ) -> typing.Tuple[ArticulationAction, typing.Optional[np.ndarray]]:
        self.object_size = source_size
        self._ensure_randomization()

        if target_end_effector_orientation is None:
            base_orient = np.radians([0, 90, 10])
        else:
            # Convert quat back to euler for noise addition, then back
            base_orient = R.from_quat(target_end_effector_orientation).as_euler('xyz')

        noisy_orient = base_orient + np.radians(self._orient_noise)
        orient_quat = R.from_euler('xyz', noisy_orient).as_quat()

        speed = (pour_speed if pour_speed is not None
                 else self._pour_default_speed) * self._speed_factor

        nv = current_joint_velocities.shape[0]

        if self._event >= len(self._events_dt):
            articulation_controller.switch_dof_control_mode(dof_index=self.wrist_dof_index, mode="velocity")
            action = ArticulationAction(joint_velocities=[None] * nv)
            return action, self._build_record_array(action, current_joint_positions)

        if self._event == 0:
            target_position[2] += self._random_height_1
            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=orient_quat)
            self._random_height_1 = self._uniform(*self._height_range_1)
            if self._xy_reached(gripper_position, target_position, threshold=0.08):
                self._next_event()
                return action, self._build_record_array(action, current_joint_positions)

        elif self._event == 1:
            target_position[0] += 0.02 + self._x_offset_noise
            target_position[2] += (self._random_height_2
                                   + self.object_size[2] / 2
                                   + self.get_pickz_offset(source_name))
            target_position[1] -= (self.object_size[2] / 2
                                   - self.get_pickz_offset(source_name))
            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=orient_quat)
            self._random_height_2 = self._uniform(*self._height_range_2)
            if self._xy_reached(gripper_position, target_position):
                self._next_event()
                return action, self._build_record_array(action, current_joint_positions)

        elif self._position_pour and self._event in (2, 3, 4, 5):
            # Position-controlled pour tilt: hold the arm at the pour-start pose
            # and ramp ONLY wrist joint 6 to a fixed angle (self._t = phase
            # progress), then back. Records the real commanded position, so
            # open-loop replay reproduces the tilt exactly (the velocity pour
            # recorded an integrated-velocity angle that drifted from the joint).
            if self._event == 2 and self._pour_arm0 is None:
                articulation_controller.switch_dof_control_mode(dof_index=self.wrist_dof_index, mode="position")
                base = (current_joint_positions[:7]
                        if current_joint_positions is not None
                        else self._last_arm_positions)
                self._pour_arm0 = (np.asarray(base, dtype=np.float64).copy()
                                   if base is not None else None)
            if self._pour_arm0 is None:
                action = self._null_action(nv)
            else:
                direction = -1.0 if speed < 0 else 1.0
                delta = direction * self._pour_angle_rad
                frac = float(np.clip(self._t, 0.0, 1.0))
                arm = self._pour_arm0.copy()
                if self._event == 2:      # ramp to full tilt
                    arm[6] = self._pour_arm0[6] + delta * frac
                elif self._event == 3:    # hold at full tilt
                    arm[6] = self._pour_arm0[6] + delta
                elif self._event == 4:    # ramp back to upright
                    arm[6] = self._pour_arm0[6] + delta * (1.0 - frac)
                else:                     # event 5: hold upright
                    arm[6] = self._pour_arm0[6]
                jp = [float(x) for x in arm[:7]] + [None] * (nv - 7)
                action = ArticulationAction(joint_positions=jp)

        elif self._event == 2:
            articulation_controller.switch_dof_control_mode(dof_index=self.wrist_dof_index, mode="velocity")
            vels = [None] * nv
            vels[6] = speed
            action = ArticulationAction(joint_velocities=vels)

        elif self._event == 3:
            articulation_controller.switch_dof_control_mode(dof_index=self.wrist_dof_index, mode="velocity")
            vels = [None] * nv
            vels[6] = 0
            action = ArticulationAction(joint_velocities=vels)

        elif self._event == 4:
            articulation_controller.switch_dof_control_mode(dof_index=self.wrist_dof_index, mode="velocity")
            vels = [None] * nv
            vels[6] = -speed
            action = ArticulationAction(joint_velocities=vels)

        elif self._event == 5:
            articulation_controller.switch_dof_control_mode(dof_index=self.wrist_dof_index, mode="velocity")
            vels = [None] * nv
            vels[6] = 0
            action = ArticulationAction(joint_velocities=vels)

        self._advance_state()
        return action, self._build_record_array(action, current_joint_positions)

    # ── Offset table ─────────────────────────────────────────────

    def get_pickz_offset(self, item_name):
        table = {
            "conical_bottle02": 0.03, "conical_bottle03": 0.07,
            "conical_bottle04": 0.08, "beaker2": 0.02,
            "graduated_cylinder_01": 0.0, "graduated_cylinder_02": 0.0,
            "graduated_cylinder_03": 0.0, "graduated_cylinder_04": 0.0,
            "volume_flask": 0.05, "beaker": 0.02, "beaker_l": 0.02,
        }
        for key, val in table.items():
            if key in item_name.lower():
                return val
        return self.object_size[2] * 2 / 5

    # ── Reset ────────────────────────────────────────────────────

    def reset(self, events_dt=None):
        super().reset(events_dt)
        self.object_size = None
        self._random_height_1 = self._uniform(*self._height_range_1)
        self._random_height_2 = self._uniform(*self._height_range_2)
        self._speed_factor = 1.0
        self._x_offset_noise = 0.0
        self._orient_noise = np.zeros(3)
        self._last_arm_positions = None
        self._pour_arm0 = None
