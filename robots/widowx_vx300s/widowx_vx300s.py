# SPDX-License-Identifier: Apache-2.0

"""Trossen/Interbotix WidowX VX300s - 6-DOF arm with a parallel gripper.

Two things differ from the ARX arms. Its URDF namespaces every link (``vx300s/...``),
which the USD importer flattens to ``vx300s_...`` prim names -- so the Lula frame name
and the USD prim path spell the same link differently. And its fingers do not close to
zero: travel is 21..57 mm per side, i.e. a 42 mm minimum opening, which is still well
inside the beaker's 68.8 mm base.
"""

import os

import carb
import numpy as np
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper

from robots.base_robot import BaseRobot
from utils.object_utils import ObjectUtils


class WidowXVX300s(BaseRobot):
    """WidowX VX300s robotic arm."""

    # 9 values, not 8: PhysX exposes the Interbotix gripper's own drive joint
    # (``gripper``, continuous) in addition to the two fingers, and set_joint_positions
    # needs one entry per articulation DOF. It is not in gripper_joint_names because the
    # gripper controller drives the fingers directly.
    # Order: 6 arm joints (home solved by IK for the middle of the workspace),
    #        gripper drive, then the two fingers.
    DEFAULT_JOINT_POSITIONS = np.array([-0.0001, 0.8240, -0.7567, 0.0003, 1.5031, -0.4363, 0.0, 0.057, -0.057])

    _ARM_JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    _GRIPPER_JOINT_NAMES = ["left_finger", "right_finger"]

    FINGER_OPEN_M = 0.057
    FINGER_CLOSED_M = 0.021  # the mechanism's hard minimum, not zero

    def __init__(
        self,
        prim_path: str = "/World/WidowX",
        name: str = "widowx_vx300s",
        usd_path: str | None = None,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        default_joint_positions: np.ndarray | None = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._default_joint_positions = (
            default_joint_positions if default_joint_positions is not None else self.DEFAULT_JOINT_POSITIONS.copy()
        )

        if not prim.IsValid():
            if usd_path is None:
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                usd_path = os.path.join(repo_root, "assets/robots/widowx_vx300s.usd")
            if not os.path.exists(usd_path):
                carb.log_error(f"Could not find WidowX USD at {usd_path}")
                raise FileNotFoundError(f"WidowX USD not found: {usd_path}")
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        self._end_effector_prim_path = prim_path + "/vx300s_gripper_link"

        super().__init__(prim_path=prim_path, name=name, position=position, orientation=orientation)

        self._gripper_open_position = np.array([self.FINGER_OPEN_M, -self.FINGER_OPEN_M])
        self._gripper_closed_position = np.array([self.FINGER_CLOSED_M, -self.FINGER_CLOSED_M])

        self._gripper = ParallelGripper(
            end_effector_prim_path=self._end_effector_prim_path,
            joint_prim_names=self._GRIPPER_JOINT_NAMES.copy(),
            joint_opened_positions=self._gripper_open_position,
            joint_closed_positions=self._gripper_closed_position,
            action_deltas=np.array([0.01, 0.01]) / get_stage_units(),
        )

    @property
    def arm_joint_names(self) -> list[str]:
        return self._ARM_JOINT_NAMES

    @property
    def gripper_joint_names(self) -> list[str]:
        return self._GRIPPER_JOINT_NAMES

    @property
    def gripper_distance_multipliers(self) -> list[float]:
        """Mirrored fingers: right_finger mimics left with multiplier -1."""
        return [1.0, -1.0]

    @property
    def end_effector_prim_path(self) -> str:
        return self._end_effector_prim_path

    @property
    def gripper_center_prim_path(self) -> str:
        """USD spelling of the URDF's ``vx300s/ee_gripper_link``."""
        return self.prim_path_str + "/vx300s_ee_gripper_link"

    @property
    def tool_frame_correction_euler_deg(self) -> list[float]:
        """Approach is local +X here, not +Z (measured by FK); fingers stay on +Y.

        Ry(-90) maps local +X onto the canonical +Z and leaves +Y untouched. Ry(+90)
        sends +X to -Z instead, which points the gripper away from the object and makes
        every grasp target unsolvable.
        """
        return [0.0, -90.0, 0.0]

    @property
    def ik_end_effector_frame(self) -> str:
        """URDF spelling of the same link -- Lula keeps the namespace slash."""
        return "vx300s/ee_gripper_link"

    @property
    def motion_config(self) -> dict:
        package_dir = os.path.dirname(os.path.abspath(__file__))
        rmpflow_dir = os.path.join(package_dir, "rmpflow")
        return {
            "end_effector_frame_name": "vx300s/ee_gripper_link",
            "maximum_substep_size": 0.00334,
            "robot_description_path": os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
            "urdf_path": os.path.join(package_dir, "widowx_vx300s.urdf"),
            "rmpflow_config_path": os.path.join(rmpflow_dir, "widowx_vx300s_rmpflow_common.yaml"),
        }

    def get_gripper_position(self) -> np.ndarray:
        return ObjectUtils.get_instance().get_object_xform_position(object_path=self.gripper_center_prim_path)

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view)
        self._end_effector = SingleRigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)

        dof_names = self.dof_names if self.dof_names is not None else self.get_all_joint_names()
        self._gripper.set_default_state(
            np.array(self._default_joint_positions[-self.num_gripper_joints :], dtype=np.float64)
        )
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=dof_names,
        )
        self.enforce_requested_world_pose()
        self.set_joint_positions(self._default_joint_positions)

    def post_reset(self) -> None:
        super().post_reset()
        self._gripper.post_reset()
        for dof_index in self._gripper.joint_dof_indicies:
            self._articulation_controller.switch_dof_control_mode(dof_index=dof_index, mode="position")
        self.set_joint_positions(self._default_joint_positions)
