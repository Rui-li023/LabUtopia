from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from scipy.spatial.transform import Rotation as R, Slerp

from .atomic_base_controller import AtomicBaseController
from robots.base_robot import BaseRobot


class OpenController(AtomicBaseController):
    """State machine for opening drawers/doors (8 phases).

    Drawer: approach → align → close gripper → pull → settle → open gripper → retreat → done.
    Door:   approach → align → close gripper → arc pull → settle → open gripper → retreat → done.

    Per-episode randomization:
      - approach offset noise (±0.015 m)
      - retreat offset noise (±0.02 m)
    """

    DEFAULT_DT = [0.0025, 0.005, 0.08, 0.002, 0.05, 0.05, 0.01, 0.008]

    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
        gripper=None,
        events_dt: typing.Optional[typing.List[float]] = None,
        furniture_type: str = "drawer",
        door_width: float = 0.3,
        door_open_direction: str = "counterclockwise",
        robot: typing.Optional[BaseRobot] = None,
    ) -> None:
        super().__init__(
            name=name,
            cspace_controller=cspace_controller,
            events_dt=events_dt,
            default_events_dt=self.DEFAULT_DT,
            robot=robot,
        )
        self._gripper = gripper
        self.furniture_type = furniture_type
        self.door_width = door_width
        self.door_open_direction = door_open_direction
        self.position_rotation_interp_iter = None
        self._position_threshold = 0.01 / get_stage_units()

        # Per-episode noise
        self._approach_noise = 0.0
        self._retreat_noise = 0.0

    # ── Randomization ────────────────────────────────────────────

    def _sample_randomization(self):
        self._approach_noise = self._noisy(0.0, 0.015)
        self._retreat_noise = self._noisy(0.0, 0.02)

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self,
        handle_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        revolute_joint_position: np.ndarray = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        angle: float = 50.0,
        close_gripper_distance: float = 0.023,
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        n = current_joint_positions.shape[0]

        # Guard: bail out with a null action if the task could not resolve
        # the handle pose (USD/task mismatch), avoiding a C++ crash.
        if handle_position is None:
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions,
                gripper_state=self._current_gripper_state)

        if self._start:
            self._start = False
            self._open_gripper()
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions,
                gripper_state=self._current_gripper_state)

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(
                [0, 110, 0], degrees=True, extrinsic=False)

        self._ensure_randomization()

        # Guard: the caller advances phases on its own success condition, so a
        # state machine that runs out of events while the phase is still active
        # keeps getting forward()ed. Indexing past _events_dt then raised
        # IndexError *inside* the Isaac callback, which takes down the whole
        # simulator with a SIGSEGV (this killed a full L4 open_transport_pour
        # collect run). Hold the last pose instead and let the task's own
        # max_steps decide the episode. Mirrors the same guard the pour atomic
        # has had (atomic_actions/pour_controller.py:200).
        if self._event >= len(self._events_dt):
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions,
                gripper_state=self._current_gripper_state)

        self._t += self._events_dt[self._event]

        if self.furniture_type == "drawer":
            action = self._drawer_phase(handle_position, end_effector_orientation,
                                        current_joint_positions, gripper_position)
        else:
            action = self._door_phase(handle_position, end_effector_orientation,
                                      current_joint_positions, revolute_joint_position,
                                      gripper_position, angle, close_gripper_distance)

        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        return action, self._build_record_array(
            action, current_joint_positions,
            gripper_state=self._current_gripper_state)

    # ── Drawer phases ────────────────────────────────────────────

    def _drawer_phase(self, handle_pos, orient, jpos, grip_pos):
        n = jpos.shape[0]
        su = get_stage_units()
        approach = 0.08 + self._approach_noise

        if self._event == 0:
            handle_pos[0] -= approach / su
            action = self._cspace_controller.forward(
                target_end_effector_position=handle_pos,
                target_end_effector_orientation=orient)
            if self._xy_reached(grip_pos, handle_pos):
                self._next_event()
            return action

        elif self._event == 1:
            handle_pos[0] -= 0.015
            action = self._cspace_controller.forward(
                target_end_effector_position=handle_pos,
                target_end_effector_orientation=orient)
            if self._xy_reached(grip_pos, handle_pos,
                                threshold=self._position_threshold / 3):
                self._next_event()
            return action

        elif self._event == 2:
            self._close_gripper()
            self.target_position = handle_pos.copy()
            self.target_position[0] -= 0.1 / su
            return self._null_action(n)

        elif self._event == 3:
            handle_pos[0] -= 0.18 / su
            return self._cspace_controller.forward(
                target_end_effector_position=handle_pos,
                target_end_effector_orientation=orient)

        elif self._event == 4:
            return self._null_action(n)

        elif self._event == 5:
            self._open_gripper()
            return self._null_action(n)

        elif self._event == 6:
            retreat = 0.12 + self._retreat_noise
            handle_pos[0] -= retreat / su
            handle_pos[2] += 0.06
            return self._cspace_controller.forward(
                target_end_effector_position=handle_pos,
                target_end_effector_orientation=orient)

        else:
            return self._null_action(n)

    # ── Door phases ──────────────────────────────────────────────

    def _door_phase(self, handle_pos, orient, jpos, rev_pos, grip_pos,
                    angle, close_dist):
        n = jpos.shape[0]
        approach = 0.08 + self._approach_noise

        if self._event == 0:
            handle_pos[0] -= approach
            action = self._cspace_controller.forward(
                target_end_effector_position=handle_pos,
                target_end_effector_orientation=orient)
            if self._xy_reached(grip_pos, handle_pos):
                self._next_event()
            return action

        elif self._event == 1:
            handle_pos[0] -= 0.015
            action = self._cspace_controller.forward(
                target_end_effector_position=handle_pos,
                target_end_effector_orientation=orient)
            if self._xy_reached(grip_pos, handle_pos,
                                threshold=self._position_threshold / 3):
                self._next_event()
            return action

        elif self._event == 2:
            handle_pos[0] -= 0.015
            self._close_gripper()
            self.start_position = handle_pos.copy()
            return self._null_action(n)

        elif self._event == 3:
            if self.position_rotation_interp_iter is None:
                a = -angle if rev_pos[1] > self.start_position[1] else angle
                target = self._rotate_point_around_z(
                    self.start_position, rev_pos, a)
                num = int(600 * np.linalg.norm(self.start_position - target))
                alphas = np.linspace(0, 1, num)[1:]
                target_orient = self._rotate_quat_around_x(orient, a)
                interp_list = self._arc_interpolation(
                    self.start_position, orient, target, target_orient,
                    alphas, rev_pos)
                self.position_rotation_interp_iter = iter(interp_list)
            try:
                self.trans_interp, self.rotation_interp = next(
                    self.position_rotation_interp_iter)
                return self._cspace_controller.forward(
                    target_end_effector_position=self.trans_interp,
                    target_end_effector_orientation=self.rotation_interp)
            except StopIteration:
                self._next_event()
                return self._cspace_controller.forward(
                    target_end_effector_position=self.trans_interp,
                    target_end_effector_orientation=self.rotation_interp)

        elif self._event == 4:
            return self._null_action(n)

        elif self._event == 5:
            self._open_gripper()
            return self._null_action(n)

        elif self._event == 6:
            retreat = 0.06 + self._retreat_noise
            pos = self.trans_interp.copy()
            pos[0] -= retreat
            offset = 0.04 if rev_pos[1] <= self.start_position[1] else -0.04
            pos[1] += offset
            action = self._cspace_controller.forward(
                target_end_effector_position=pos,
                target_end_effector_orientation=self.rotation_interp)
            if self._xy_reached(grip_pos, pos):
                self._next_event()
            return action

        else:
            return self._null_action(n)

    # ── Geometry helpers ─────────────────────────────────────────

    @staticmethod
    def _rotate_quat_around_x(q, angle_deg):
        rad = np.deg2rad(angle_deg)
        q_rot = np.array([-np.sin(rad/2), 0, 0, np.cos(rad/2)])
        return (R.from_quat(q) * R.from_quat(q_rot)).as_quat()

    @staticmethod
    def _arc_interpolation(start_pos, start_quat, end_pos, end_quat,
                           alphas, pivot):
        """Interpolate along an arc around *pivot* with Slerp for rotation."""
        sq_start = start_quat[[1,2,3,0]]
        sq_end = end_quat[[1,2,3,0]]
        key_rots = R.from_quat(np.stack([sq_start, sq_end]))
        slerp = Slerp([0, 1], key_rots)
        interp_rots = slerp(alphas).as_quat()[:, [3,0,1,2]]  # back to wxyz

        r0 = start_pos[:2] - pivot[:2]
        r1 = end_pos[:2] - pivot[:2]
        rad0, rad1 = np.linalg.norm(r0), np.linalg.norm(r1)
        radii = np.linspace(rad0, rad1, len(alphas) + 1)[1:]
        t0 = np.arctan2(r0[1], r0[0])
        t1 = np.arctan2(r1[1], r1[0])
        dt = t1 - t0
        if dt > np.pi:
            dt -= 2 * np.pi
        elif dt < -np.pi:
            dt += 2 * np.pi
        dt = np.clip(dt, -np.pi/2, np.pi/2)
        thetas = np.linspace(t0, t0 + dt, len(alphas) + 1)[1:]

        result = []
        for a, radius, theta, rot in zip(alphas, radii, thetas, interp_rots):
            pos = np.array([
                pivot[0] + radius * np.cos(theta),
                pivot[1] + radius * np.sin(theta),
                a * end_pos[2] + (1 - a) * start_pos[2],
            ])
            result.append((pos, rot))
        return result

    # ── Reset ────────────────────────────────────────────────────

    def reset(self):
        super().reset()
        self.position_rotation_interp_iter = None
        self._approach_noise = 0.0
        self._retreat_noise = 0.0
