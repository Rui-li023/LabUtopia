# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Optional

import carb
import numpy as np
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from isaacsim.sensors.physics import ContactSensor
from isaacsim.sensors.camera import Camera

from utils.object_utils import ObjectUtils


class Piper(Robot):
    """Agilex Piper Robot

    6-DOF robotic arm with parallel gripper.

    Args:
        prim_path (str): USD prim path for the robot.
        name (str, optional): Robot name. Defaults to "piper".
        usd_path (Optional[str], optional): Path to USD file. Defaults to None.
        position (Optional[np.ndarray], optional): Robot base position. Defaults to None.
        orientation (Optional[np.ndarray], optional): Robot base orientation. Defaults to None.
        end_effector_prim_name (Optional[str], optional): End effector prim name. Defaults to None.
        gripper_dof_names (Optional[List[str]], optional): Gripper joint names. Defaults to None.
        gripper_open_position (Optional[np.ndarray], optional): Gripper open position. Defaults to None.
        gripper_closed_position (Optional[np.ndarray], optional): Gripper closed position. Defaults to None.
        deltas (Optional[np.ndarray], optional): Gripper action deltas. Defaults to None.
        default_joint_positions (Optional[np.ndarray], optional): Default joint positions. Defaults to None.
    """

    # Default home position: 6 arm joints + 2 gripper joints
    DEFAULT_JOINT_POSITIONS = np.array([0.0, 1.57, -1.57, 0.0, 0.0, 0.0, 0.035, -0.035])

    def __init__(
        self,
        prim_path: str = "/World/Piper",
        name: str = "piper",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        deltas: Optional[np.ndarray] = None,
        default_joint_positions: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        self.prim_path_str = prim_path
        self._default_joint_positions = (
            default_joint_positions if default_joint_positions is not None
            else self.DEFAULT_JOINT_POSITIONS.copy()
        )

        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                usd_path = os.path.join(current_dir, "piper.usd")
                if not os.path.exists(usd_path):
                    carb.log_error(f"Could not find Piper USD file at {usd_path}")
                    raise FileNotFoundError(f"Piper USD file not found: {usd_path}")
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        # Set default end effector path
        if self._end_effector_prim_name is None:
            self._end_effector_prim_path = prim_path + "/gripper_base"
        else:
            self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name

        # Default gripper configuration: 2 gripper joints (joint7, joint8)
        if gripper_dof_names is None:
            gripper_dof_names = ["joint7", "joint8"]
        if gripper_open_position is None:
            gripper_open_position = np.array([0.035, -0.035])
        if gripper_closed_position is None:
            gripper_closed_position = np.array([0.0, 0.0])

        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )

        if deltas is None:
            deltas = np.array([0.01, 0.01]) / get_stage_units()
        self._gripper = ParallelGripper(
            end_effector_prim_path=self._end_effector_prim_path,
            joint_prim_names=gripper_dof_names,
            joint_opened_positions=gripper_open_position,
            joint_closed_positions=gripper_closed_position,
            action_deltas=deltas,
        )

        # Contact sensors for gripper fingers
        self.left_contact_sensor = ContactSensor(
            prim_path=prim_path + "/link7" + "/contact_sensor",
            name="contact_sensor_left",
            min_threshold=0,
            max_threshold=10000000,
            radius=0.1,
        )

        self.right_contact_sensor = ContactSensor(
            prim_path=prim_path + "/link8" + "/contact_sensor",
            name="contact_sensor_right",
            min_threshold=0,
            max_threshold=10000000,
            radius=0.1,
        )

        # Wrist camera mounted on gripper base
        self.camera = Camera(
            prim_path=prim_path + "/gripper_base/arm_camera",
            translation=np.array([-0.5, 0.0, -0.1]),
            frequency=60,
            resolution=(256, 256),
            orientation=np.array([0.20083, 0.67799, -0.67799, -0.20083]),
        )
        self.camera.set_local_pose(
            translation=np.array([-0.5, 0.0, -0.1]),
            orientation=np.array([0.20083, 0.67799, -0.67799, -0.20083]),
            camera_axes="usd"
        )
        self.camera.set_clipping_range(near_distance=0.01)
        self.camera.set_focal_length(1.)
        return

    def get_contact_sensor(self):
        """Get contact sensors.

        Returns:
            tuple: (left_contact_sensor, right_contact_sensor)
        """
        return self.left_contact_sensor, self.right_contact_sensor

    @property
    def end_effector(self) -> SingleRigidPrim:
        """Get end effector rigid body.

        Returns:
            SingleRigidPrim: End effector prim.
        """
        return self._end_effector

    @property
    def gripper(self) -> ParallelGripper:
        """Get gripper controller.

        Returns:
            ParallelGripper: Gripper controller.
        """
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        """Initialize robot."""
        super().initialize(physics_sim_view)
        self._end_effector = SingleRigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)

        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        self.set_joint_positions(self._default_joint_positions)
        return

    def post_reset(self) -> None:
        """Post reset callback."""
        super().post_reset()
        self._gripper.post_reset()
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[1], mode="position"
        )
        self.set_joint_positions(self._default_joint_positions)
        return

    def get_gripper_position(self) -> np.ndarray:
        """Get gripper position in world coordinates.

        Returns:
            np.ndarray: Gripper position [x, y, z].
        """
        return ObjectUtils.get_instance().get_object_xform_position(
            object_path=self.prim_path_str + "/gripper_base"
        )
