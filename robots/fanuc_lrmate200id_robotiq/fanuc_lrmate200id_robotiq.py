# SPDX-License-Identifier: Apache-2.0

"""FANUC LR Mate 200iD with a Robotiq 2F-85 gripper."""

import os

import numpy as np

from robots.ur5e_robotiq.ur5e_robotiq import UR5eRobotiq


class FanucLRMate200iDRobotiq(UR5eRobotiq):
    """Six-axis LR Mate using the shared physical Robotiq articulation."""

    # RMPFlow-converged high pre-grasp pose at the centre of the configured
    # workspace. The vendor zero/showroom pose leaves the TCP at z~=1.7 m and
    # consumes the whole first phase just folding down toward the bench.
    DEFAULT_JOINT_POSITIONS = np.array([-0.103, 0.576, -0.803, 0.001, 1.375, 3.141] + [0.0] * 6)
    _ARM_JOINT_NAMES = [f"joint_{index}" for index in range(1, 7)]

    def __init__(
        self,
        prim_path: str = "/World/FanucLRMate200iDRobotiq",
        name: str = "fanuc_lrmate200id_robotiq",
        usd_path: str | None = None,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        default_joint_positions: np.ndarray | None = None,
    ) -> None:
        if usd_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            usd_path = os.path.join(repo_root, "assets/robots/fanuc_lrmate200id_robotiq85.usd")
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
            "urdf_path": os.path.join(package_dir, "fanuc_lrmate200id_robotiq85.urdf"),
            "rmpflow_config_path": os.path.join(rmpflow_dir, "fanuc_lrmate200id_robotiq_rmpflow_common.yaml"),
        }

    @property
    def gripper_center_prim_path(self) -> str:
        return self.prim_path_str + "/grasp_frame"

    @property
    def ik_end_effector_frame(self) -> str:
        return "grasp_frame"
