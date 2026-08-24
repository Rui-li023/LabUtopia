# SPDX-License-Identifier: Apache-2.0

"""Universal Robots UR3e with a Robotiq 2F-85 gripper."""

import os

import numpy as np

from robots.ur5e_robotiq.ur5e_robotiq import UR5eRobotiq


class UR3eRobotiq(UR5eRobotiq):
    """UR3e + Robotiq 2F-85.

    The UR3e and UR5e share the same six joint names and Robotiq attachment frame.
    Keeping the gripper implementation in :class:`UR5eRobotiq` makes the driver and
    mimic-joint mapping identical; only the arm asset and its Lula files differ.
    """

    DEFAULT_JOINT_POSITIONS = np.array([0.0, -1.2, 1.4, -1.75, -1.57, 0.0] + [0.0] * 6)

    def __init__(
        self,
        prim_path: str = "/World/UR3eRobotiq",
        name: str = "ur3e_robotiq",
        usd_path: str | None = None,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        default_joint_positions: np.ndarray | None = None,
    ) -> None:
        if usd_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            usd_path = os.path.join(repo_root, "assets/robots/ur3e_robotiq85.usd")
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
            "urdf_path": os.path.join(package_dir, "ur3e_robotiq85.urdf"),
            "rmpflow_config_path": os.path.join(rmpflow_dir, "ur3e_robotiq_rmpflow_common.yaml"),
        }
