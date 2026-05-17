from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from scipy.spatial.transform import Rotation as R, Slerp

from .atomic_base_controller import AtomicBaseController
from robots.base_robot import GRIPPER_OPEN, GRIPPER_CLOSED


class CloseController(AtomicBaseController):
    """State machine for closing drawers (4 phases) / doors (3 phases) / lids (3 phases).

    Drawer: approach → push → retreat → done.
    Door:   approach → arc push → retreat.
    Lid:    approach above → push down → retreat upward.

    Per-episode randomization:
      - approach offset noise (±0.015 m)
      - push force noise (via offset, ±0.01 m)
    """

    DRAWER_DEFAULT_DT = [0.0005, 0.002, 0.05, 0.008]
    DOOR_DEFAULT_DT = [0.0025, 0.005, 0.005]
    LID_DEFAULT_DT = [0.002, 0.003, 0.003, 0.008]

    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
        gripper=None,
        events_dt: typing.Optional[typing.List[float]] = None,
        furniture_type: str = "drawer",
        door_width: float = 0.3,
        door_open_direction: str = None,
        robot=None,
    ) -> None:
        if furniture_type == "drawer":
            default = self.DRAWER_DEFAULT_DT
        elif furniture_type == "lid":
            default = self.LID_DEFAULT_DT
        else:
            default = self.DOOR_DEFAULT_DT
        super().__init__(
            name=name,
            cspace_controller=cspace_controller,
            events_dt=events_dt,
            default_events_dt=default,
            robot=robot,
        )
        self.furniture_type = furniture_type
        self.door_width = door_width
        self.door_open_direction = door_open_direction
        self.position_rotation_interp_iter = None
        self.init_handle_position = None
        self._position_threshold = 0.01 / get_stage_units()

        # Per-episode noise
        self._approach_noise = 0.0
        self._push_noise = 0.0

    # ── Randomization ────────────────────────────────────────────

    def _sample_randomization(self):
        self._approach_noise = self._noisy(0.0, 0.015)
        # Positive-only push noise: never push less than the default,
        # since the downstream success check requires a minimum lid travel.
        self._push_noise = self._uniform(0.0, 0.01)

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self,
        handle_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        angle: float = 50.0,
        revolute_joint_position: np.ndarray = None,
        push_distance: float = None,
        after_move_distance: float = None,
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(
                [0, 110, 0], degrees=True, extrinsic=False)

        n = current_joint_positions.shape[0]
        # Guard: bail out with a null action if the task could not resolve
        # the handle pose (USD/task mismatch), avoiding a C++ crash.
        if handle_position is None:
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions,
                gripper_state=self._current_gripper_state)

        if self.init_handle_position is None:
            self.init_handle_position = handle_position.copy()

        self._ensure_randomization()
        self._t += self._events_dt[self._event]

        if self.furniture_type == "drawer":
            action = self._drawer_phase(
                handle_position, end_effector_orientation,
                current_joint_positions, gripper_position,
                push_distance, after_move_distance)
        elif self.furniture_type == "lid":
            action = self._lid_phase(
                handle_position, end_effector_orientation,
                current_joint_positions, gripper_position,
                push_distance, after_move_distance)
        else:
            action = self._door_phase(
                handle_position, end_effector_orientation,
                current_joint_positions, revolute_joint_position,
                gripper_position, angle, after_move_distance)

        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        return action, self._build_record_array(
            action, current_joint_positions,
            gripper_state=self._current_gripper_state)

    # ── Drawer phases ────────────────────────────────────────────

    def _drawer_phase(self, handle_pos, orient, jpos, grip_pos,
                      push_distance, after_move_distance):
        n = jpos.shape[0]
        su = get_stage_units()
        approach = 0.1 + self._approach_noise

        if self._event == 0:
            target = handle_pos.copy()
            target[0] -= approach / su
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient)
            if self._xy_reached(grip_pos, target):
                self._next_event()
            return action

        elif self._event == 1:
            target = handle_pos.copy()
            target[0] += (0.05 + self._push_noise) / su
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient)
            if (push_distance is not None
                    and np.linalg.norm(handle_pos - self.init_handle_position) > push_distance):
                self._next_event()
            return action

        elif self._event == 2:
            target = handle_pos.copy()
            target[0] -= approach / su
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient)
            if float(np.linalg.norm(grip_pos[:2] - handle_pos[:2])) > 0.05:
                self._next_event()
            return action

        else:
            return self._null_action(n)

    # ── Lid phases (top-to-bottom arc close) ────────────────────

    def _lid_phase(self, handle_pos, orient, jpos, grip_pos,
                   push_distance, after_move_distance):
        """Close a lid by tracing a downward arc (no gripping needed).

        Phase 0: Lift — move high above the door to clear it.
        Phase 1: Cross — move over to the far side of the door.
        Phase 2: Arc push — push the lid down in an arc.
        Phase 3: Retreat — move upward and away.
        """
        n = jpos.shape[0]
        approach = 0.08 + self._approach_noise

        if self._event == 0:
            # Lift: move high above the door to clear the open lid
            target = handle_pos.copy()
            target[1] += 0.1
            target[2] += 0.25 + approach
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient)
            if self._xyz_reached(grip_pos, target, threshold=0.02):
                self._next_event()
            return action

        elif self._event == 1:
            # Cross: move to the far side of the door (past it in X)
            target = handle_pos.copy()
            target[0] += 0.10
            target[1] += 0.25
            target[2] += approach
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient)
            if self._xyz_reached(grip_pos, target, threshold=0.02):
                self._next_event()
            return action

        elif self._event == 2:
            # Arc push: trace a downward arc to push the lid closed
            if self.position_rotation_interp_iter is None:
                self.start_position = grip_pos.copy()
                end_position = self.init_handle_position.copy()
                end_position[2] -= (0.05 + self._push_noise)
                num_steps = max(int(400 * np.linalg.norm(
                    self.start_position - end_position)), 10)
                alphas = np.linspace(0, 1, num_steps)[1:]
                interp_list = []
                for a in alphas:
                    pos = (1 - a) * self.start_position + a * end_position
                    # Arc curvature: push toward the hinge side
                    pos[0] -= np.sin(a * np.pi) * 0.02
                    interp_list.append(pos)
                self.position_rotation_interp_iter = iter(interp_list)
            try:
                target = next(self.position_rotation_interp_iter)
                return self._cspace_controller.forward(
                    target_end_effector_position=target,
                    target_end_effector_orientation=orient)
            except StopIteration:
                self._next_event()
                return self._null_action(n)

        elif self._event == 3:
            # Retreat: move upward and away
            target = handle_pos.copy()
            target[2] += (after_move_distance if after_move_distance else 0.15)
            target[0] -= 0.1
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient)
            if self._xyz_reached(grip_pos, target, threshold=0.03):
                self._next_event()
            return action

        else:
            return self._null_action(n)

    # ── Door phases ──────────────────────────────────────────────

    def _door_phase(self, handle_pos, orient, jpos, rev_pos, grip_pos,
                    angle, after_move_distance):
        n = jpos.shape[0]
        approach = 0.05 + self._approach_noise

        if self._event == 0:
            handle_pos[0] -= approach
            handle_pos[2] += 0.1
            action = self._cspace_controller.forward(
                target_end_effector_position=handle_pos,
                target_end_effector_orientation=orient)
            if self._xy_reached(grip_pos, handle_pos):
                self._next_event()
            return action

        elif self._event == 1:
            if self.position_rotation_interp_iter is None:
                self.start_position = handle_pos.copy()
                a = -angle if rev_pos[1] > self.start_position[1] else angle
                target = self._rotate_point_around_z(
                    self.start_position, rev_pos, -a)
                num = int(600 * np.linalg.norm(self.start_position - target))
                alphas = np.linspace(0, 1, num)[1:]
                target_orient = self._rotate_quat_around_x(orient, -a)
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
                return self._null_action(n)

        elif self._event == 2:
            target = handle_pos.copy()
            target[0] -= after_move_distance
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=self.rotation_interp)
            if self._xy_reached(grip_pos, target, threshold=0.02):
                self._next_event()
            return action

        else:
            return self._null_action(n)

    # ── Geometry helpers (shared with OpenController) ─────────────

    @staticmethod
    def _rotate_quat_around_x(q, angle_deg):
        rad = np.deg2rad(angle_deg)
        q_rot = np.array([-np.sin(rad/2), 0, 0, np.cos(rad/2)])
        return (R.from_quat(q) * R.from_quat(q_rot)).as_quat()

    @staticmethod
    def _arc_interpolation(start_pos, start_quat, end_pos, end_quat,
                           alphas, pivot):
        sq_start = start_quat[[1,2,3,0]]
        sq_end = end_quat[[1,2,3,0]]
        key_rots = R.from_quat(np.stack([sq_start, sq_end]))
        slerp = Slerp([0, 1], key_rots)
        interp_rots = slerp(alphas).as_quat()[:, [3,0,1,2]]

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
        self.init_handle_position = None
        self._approach_noise = 0.0
        self._push_noise = 0.0
