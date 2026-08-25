# SPDX-License-Identifier: Apache-2.0

"""UFACTORY xArm7 with a Robotiq 2F-85 gripper."""

import os

import numpy as np

from robots.ur5e_robotiq.ur5e_robotiq import UR5eRobotiq


class XArm7Robotiq(UR5eRobotiq):
    """xArm7 + Robotiq 2F-85.

    The Robotiq driver and five mimic joints are the same revolute layout used by
    the UR/Robotiq adapters.  Only the seven-joint arm chain and asset/Lula paths
    differ, so keeping the gripper implementation inherited avoids a second set
    of width-to-angle semantics.
    """

    # joint4 is limited to [-0.19198, 3.927] rad by the xArm URDF.  The old
    # -0.6 rad preview/default pose was outside that range, so PhysX clamped
    # the real arm while Lula kept its invalid nullspace seed; joint4 then
    # accumulated a ~0.7 rad tracking error on every pick.
    DEFAULT_JOINT_POSITIONS = np.array([0.0, -0.6, 0.0, 0.6, 0.0, 0.9, 0.0] + [0.0] * 6)

    _ARM_JOINT_NAMES = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]

    def __init__(
        self,
        prim_path: str = "/World/XArm7Robotiq",
        name: str = "xarm7_robotiq",
        usd_path: str | None = None,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        default_joint_positions: np.ndarray | None = None,
    ) -> None:
        if usd_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            usd_path = os.path.join(repo_root, "assets/robots/xarm7_robotiq85.usd")
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
            "urdf_path": os.path.join(package_dir, "xarm7_robotiq85.urdf"),
            "rmpflow_config_path": os.path.join(rmpflow_dir, "xarm7_robotiq_rmpflow_common.yaml"),
        }
