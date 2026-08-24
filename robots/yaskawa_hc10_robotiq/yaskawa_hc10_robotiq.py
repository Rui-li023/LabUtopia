# SPDX-License-Identifier: Apache-2.0

"""Yaskawa Motoman HC10 with a Robotiq 2F-85 gripper."""

import os

import numpy as np

from robots.ur5e_robotiq.ur5e_robotiq import UR5eRobotiq


class YaskawaHC10Robotiq(UR5eRobotiq):
    """Six-axis HC10 using the shared physical Robotiq articulation."""

    DEFAULT_JOINT_POSITIONS = np.array([0.064, 0.525, 1.536, 1.495, -0.477, 1.353] + [0.0] * 6)
    _ARM_JOINT_NAMES = [
        "joint_1_s",
        "joint_2_l",
        "joint_3_u",
        "joint_4_r",
        "joint_5_b",
        "joint_6_t",
    ]

    def __init__(
        self,
        prim_path: str = "/World/YaskawaHC10Robotiq",
        name: str = "yaskawa_hc10_robotiq",
        usd_path: str | None = None,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        default_joint_positions: np.ndarray | None = None,
    ) -> None:
        if usd_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            usd_path = os.path.join(repo_root, "assets/robots/yaskawa_hc10_robotiq85.usd")
        super().__init__(
            prim_path=prim_path,
            name=name,
            usd_path=usd_path,
            position=position,
            orientation=orientation,
            default_joint_positions=default_joint_positions,
        )

    @property
    def motion_config(self) -> dict:
        package_dir = os.path.dirname(os.path.abspath(__file__))
        rmpflow_dir = os.path.join(package_dir, "rmpflow")
        return {
            "end_effector_frame_name": "grasp_frame",
            "maximum_substep_size": 0.00334,
            "robot_description_path": os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
            "urdf_path": os.path.join(package_dir, "yaskawa_hc10_robotiq85.urdf"),
            "rmpflow_config_path": os.path.join(rmpflow_dir, "yaskawa_hc10_robotiq_rmpflow_common.yaml"),
        }

    @property
    def gripper_center_prim_path(self) -> str:
        return self.prim_path_str + "/grasp_frame"

    @property
    def ik_end_effector_frame(self) -> str:
        return "grasp_frame"
