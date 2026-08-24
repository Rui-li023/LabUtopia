# SPDX-License-Identifier: Apache-2.0

"""UFACTORY xArm6 with a Robotiq 2F-85 gripper."""

import os

import numpy as np

from robots.xarm7_robotiq.xarm7_robotiq import XArm7Robotiq


class XArm6Robotiq(XArm7Robotiq):
    """Six-axis xArm using the same physical 2F-85 driver as the xArm7."""

    DEFAULT_JOINT_POSITIONS = np.array([0.0, -0.6, -0.6, 0.0, 0.9, 0.0] + [0.0] * 6)
    _ARM_JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]

    def __init__(
        self,
        prim_path: str = "/World/XArm6Robotiq",
        name: str = "xarm6_robotiq",
        usd_path: str | None = None,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        default_joint_positions: np.ndarray | None = None,
    ) -> None:
        if usd_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            usd_path = os.path.join(repo_root, "assets/robots/xarm6_robotiq85.usd")
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
            "urdf_path": os.path.join(package_dir, "xarm6_robotiq85.urdf"),
            "rmpflow_config_path": os.path.join(rmpflow_dir, "xarm6_robotiq_rmpflow_common.yaml"),
        }
