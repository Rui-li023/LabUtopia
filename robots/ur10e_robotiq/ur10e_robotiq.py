# SPDX-License-Identifier: Apache-2.0

"""Universal Robots UR10e with a Robotiq 2F-85 gripper."""

import os

import numpy as np

from robots.ur5e_robotiq.ur5e_robotiq import UR5eRobotiq


class UR10eRobotiq(UR5eRobotiq):
    """UR10e + Robotiq 2F-85.

    UR10e shares the UR5e's six arm joint names and the Robotiq attachment and
    driver layout.  The inherited implementation keeps the gripper semantics
    identical while this class points Lula and the asset loader at UR10e files.
    """

    DEFAULT_JOINT_POSITIONS = np.array([0.0, -1.2, 1.4, -1.75, -1.57, 0.0] + [0.0] * 6)

    def __init__(
        self,
        prim_path: str = "/World/UR10eRobotiq",
        name: str = "ur10e_robotiq",
        usd_path: str | None = None,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        default_joint_positions: np.ndarray | None = None,
    ) -> None:
        if usd_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            usd_path = os.path.join(repo_root, "assets/robots/ur10e_robotiq85.usd")
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
            "end_effector_frame_name": "tool_frame",
            "maximum_substep_size": 0.00334,
            "robot_description_path": os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
            "urdf_path": os.path.join(package_dir, "ur10e_robotiq85.urdf"),
            "rmpflow_config_path": os.path.join(rmpflow_dir, "ur10e_robotiq_rmpflow_common.yaml"),
        }
