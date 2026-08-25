# SPDX-License-Identifier: Apache-2.0

"""Kinova Gen3 7-DoF arm with a Robotiq 2F-85 gripper."""

import os

import numpy as np

from robots.ur5e_robotiq.ur5e_robotiq import UR5eRobotiq


class Gen3Robotiq(UR5eRobotiq):
    """Gen3 + Robotiq 2F-85.

    The arm-specific kinematics differ, while the Robotiq driver and mimic-joint
    interface intentionally match the UR/xArm adapters.
    """

    DEFAULT_JOINT_POSITIONS = np.array([1.57, -0.35, 3.14, -2.00, 0.0, -1.00, 1.57] + [0.0] * 6)

    GRIPPER_BASE_LINK_NAME = "gripper_base"
    DRIVE_DAMPING_RATIO = 0.05
    _ARM_JOINT_NAMES = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
    ]
    _GRIPPER_JOINT_NAMES = ["gripper_joint"]
    GRIPPER_CLOSED_RAD = 0.81

    def __init__(
        self,
        prim_path: str = "/World/Gen3Robotiq",
        name: str = "gen3_robotiq",
        usd_path: str | None = None,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        default_joint_positions: np.ndarray | None = None,
    ) -> None:
        if usd_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            usd_path = os.path.join(repo_root, "assets/robots/gen3_robotiq85.usd")
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
            "urdf_path": os.path.join(package_dir, "gen3_robotiq85.urdf"),
            "rmpflow_config_path": os.path.join(rmpflow_dir, "gen3_robotiq_rmpflow_common.yaml"),
        }

    @property
    def tool_frame_correction_euler_deg(self) -> list[float]:
        """The controlled grasp frame already uses the canonical Robotiq axes.

        The fixed +90 degree flange rotation is part of the URDF transform from
        ``tool_frame`` to ``grasp_frame``. RMPFlow targets ``grasp_frame`` directly,
        whose local +Y separates the fingers and local +Z is the approach axis, so
        applying the flange rotation here again would rotate the commanded grasp by
        another 90 degrees.
        """
        return [0.0, 0.0, 0.0]

    @property
    def gripper_center_prim_path(self) -> str:
        return self.prim_path_str + "/grasp_frame"

    @property
    def ik_end_effector_frame(self) -> str:
        return "grasp_frame"
