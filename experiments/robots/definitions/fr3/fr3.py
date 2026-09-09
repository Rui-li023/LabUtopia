# SPDX-License-Identifier: Apache-2.0

"""Franka Research 3 with its native two-finger hand."""

import os

import carb
import numpy as np
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper

from robots.base_robot import BaseRobot
from utils.object_utils import ObjectUtils


class FR3(BaseRobot):
    """Seven-axis FR3 using the upstream Franka Hand articulation."""

    # The upstream URDF declares damping=0.003 on every arm joint. That is nonzero
    # but negligible beside the imported position-drive stiffness, so the generic
    # zero-only repair would let measured-velocity feedback diverge immediately.
    ENFORCE_DRIVE_DAMPING_FLOOR = True

    DEFAULT_JOINT_POSITIONS = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398, 0.04, 0.04])

    _ARM_JOINT_NAMES = [f"fr3_joint{index}" for index in range(1, 8)]
    _GRIPPER_JOINT_NAMES = ["fr3_finger_joint1", "fr3_finger_joint2"]

    def __init__(
        self,
        prim_path: str = "/World/FR3",
        name: str = "fr3",
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
                usd_path = os.path.join(repo_root, "assets/robots/fr3.usd")
            if not os.path.exists(usd_path):
                carb.log_error(f"Could not find FR3 USD at {usd_path}")
                raise FileNotFoundError(f"FR3 USD not found: {usd_path}")
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        self._end_effector_prim_path = prim_path + "/fr3_hand"
        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
        )

        self._gripper_open_position = np.array([0.04, 0.04])
        self._gripper_closed_position = np.array([0.0, 0.0])
        self._gripper = ParallelGripper(
            end_effector_prim_path=self._end_effector_prim_path,
            joint_prim_names=self._GRIPPER_JOINT_NAMES.copy(),
            joint_opened_positions=self._gripper_open_position,
            joint_closed_positions=self._gripper_closed_position,
            action_deltas=np.array([0.04, 0.04]) / get_stage_units(),
        )

    @property
    def arm_joint_names(self) -> list[str]:
        return self._ARM_JOINT_NAMES

    @property
    def gripper_joint_names(self) -> list[str]:
        return self._GRIPPER_JOINT_NAMES

    @property
    def end_effector_prim_path(self) -> str:
        return self._end_effector_prim_path

    @property
    def gripper_center_prim_path(self) -> str:
        return self.prim_path_str + "/fr3_hand_tcp"

    @property
    def ik_end_effector_frame(self) -> str:
        return "fr3_hand_tcp"

    @property
    def motion_config(self) -> dict:
        package_dir = os.path.dirname(os.path.abspath(__file__))
        rmpflow_dir = os.path.join(package_dir, "rmpflow")
        return {
            "end_effector_frame_name": "fr3_hand_tcp",
            "maximum_substep_size": 0.00334,
            "robot_description_path": os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
            "urdf_path": os.path.join(package_dir, "fr3.urdf"),
            "rmpflow_config_path": os.path.join(rmpflow_dir, "fr3_rmpflow_common.yaml"),
        }

    def get_gripper_position(self) -> np.ndarray:
        return ObjectUtils.get_instance().get_object_xform_position(object_path=self.gripper_center_prim_path)

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view)
        self._end_effector = SingleRigidPrim(
            prim_path=self._end_effector_prim_path,
            name=self.name + "_end_effector",
        )
        self._end_effector.initialize(physics_sim_view)

        dof_names = self.dof_names if self.dof_names is not None else self.get_all_joint_names()
        self._gripper.set_default_state(self._default_joint_positions[-2:])
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
