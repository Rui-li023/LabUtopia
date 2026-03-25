# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import carb
import numpy as np
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.physics import ContactSensor
from isaacsim.sensors.camera import Camera

from robots.base_robot import BaseRobot
from utils.object_utils import ObjectUtils


class Franka(BaseRobot):
    """Franka Panda robot arm with parallel gripper.

    A 7-DOF robotic arm with a 2-finger parallel gripper, commonly used for
    manipulation tasks.

    Args:
        prim_path: USD prim path for the robot.
        name: Robot name. Defaults to "franka".
        usd_path: Path to USD file. Defaults to None (uses Isaac Sim assets).
        position: Robot base position. Defaults to None.
        orientation: Robot base orientation. Defaults to None.
        end_effector_prim_name: End effector prim name. Defaults to None.
        gripper_dof_names: Gripper joint names. Defaults to None.
        gripper_open_position: Gripper open position. Defaults to None.
        gripper_closed_position: Gripper closed position. Defaults to None.
        deltas: Gripper action deltas. Defaults to None.
        default_joint_positions: Default joint positions. Defaults to None.
    """

    # Standard Franka Panda home position: arm in a natural upright-ready pose
    DEFAULT_JOINT_POSITIONS = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398, 0.04, 0.04])

    # Franka-specific joint names
    _ARM_JOINT_NAMES = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7"
    ]
    _GRIPPER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]

    def __init__(
        self,
        prim_path: str = "/World/Franka",
        name: str = "franka",
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
        self._end_effector_prim_name = end_effector_prim_name
        self._default_joint_positions = (
            default_joint_positions if default_joint_positions is not None
            else self.DEFAULT_JOINT_POSITIONS.copy()
        )

        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                usd_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/panda_rightfinger"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"]
            if gripper_open_position is None:
                gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
            if gripper_closed_position is None:
                gripper_closed_position = np.array([0.0, 0.0])
        else:
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/panda_rightfinger"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"]
            if gripper_open_position is None:
                gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
            if gripper_closed_position is None:
                gripper_closed_position = np.array([0.0, 0.0])

        # Initialize base robot
        super().__init__(prim_path=prim_path, name=name, position=position, orientation=orientation)

        # Store gripper configuration for later initialization
        self._gripper_dof_names = gripper_dof_names
        self._gripper_open_position = gripper_open_position
        self._gripper_closed_position = gripper_closed_position
        self._deltas = deltas

        if deltas is None:
            deltas = np.array([0.05, 0.05]) / get_stage_units()
        self._gripper = ParallelGripper(
            end_effector_prim_path=self._end_effector_prim_path,
            joint_prim_names=gripper_dof_names,
            joint_opened_positions=gripper_open_position,
            joint_closed_positions=gripper_closed_position,
            action_deltas=deltas,
        )

        # Contact sensors for gripper fingers
        self.left_contact_sensor = ContactSensor(
            prim_path=prim_path + "/panda_leftfinger" + "/contact_sensor",
            name="contact_sensor_{}".format(1),
            min_threshold=0,
            max_threshold=10000000,
            radius=0.1,
        )

        self.right_contact_sensor = ContactSensor(
            prim_path=prim_path + "/panda_rightfinger" + "/contact_sensor",
            name="contact_sensor_{}".format(0),
            min_threshold=0,
            max_threshold=10000000,
            radius=0.1,
        )

        # Wrist camera mounted on panda_hand
        self._camera = Camera(
            prim_path=prim_path + "/panda_hand/arm_camera",
            translation=np.array([-0.2, -0, -0.02]),
            frequency=60,
            resolution=(256, 256),
            orientation=np.array([0.20083, 0.67799, -0.67799, -0.20083]),
        )
        self._camera.set_local_pose(orientation=np.array([0.20083, 0.67799, -0.67799, -0.20083]), camera_axes="usd")
        self._camera.set_clipping_range(near_distance=0.05)
        self._camera.set_focal_length(1.)

    # ── Implement abstract properties from BaseRobot ─────────────────────────

    @property
    def arm_joint_names(self) -> List[str]:
        """Ordered list of arm joint names."""
        return self._ARM_JOINT_NAMES

    @property
    def gripper_joint_names(self) -> List[str]:
        """Ordered list of gripper joint names."""
        return self._GRIPPER_JOINT_NAMES

    @property
    def gripper_distance_multipliers(self) -> List[float]:
        """Map scalar gripper distance to Franka finger joint directions."""
        return [1.0, 1.0]

    @property
    def end_effector_prim_path(self) -> str:
        """USD prim path of the end effector."""
        return self._end_effector_prim_path

    @property
    def gripper_center_prim_path(self) -> str:
        """USD prim path of the gripper center (tool center point)."""
        return self.prim_path_str + "/panda_hand/tool_center"

    @property
    def camera(self) -> Optional[Camera]:
        """Wrist-mounted camera."""
        return self._camera

    # ── Override get_contact_sensor ─────────────────────────────────────────

    def get_contact_sensor(self) -> Tuple[ContactSensor, ContactSensor]:
        """Get contact sensors for gripper fingers.

        Returns:
            Tuple of (left_contact_sensor, right_contact_sensor).
        """
        return self.left_contact_sensor, self.right_contact_sensor

    # ── Implement abstract methods from BaseRobot ───────────────────────────

    def get_gripper_position(self) -> np.ndarray:
        """Get gripper position in world coordinates.

        Returns:
            np.ndarray: Gripper position [x, y, z].
        """
        return ObjectUtils.get_instance().get_object_xform_position(
            object_path=self.gripper_center_prim_path
        )

    def initialize(self, physics_sim_view=None) -> None:
        """Initialize robot components."""
        super().initialize(physics_sim_view)
        self._end_effector = SingleRigidPrim(
            prim_path=self._end_effector_prim_path,
            name=self.name + "_end_effector"
        )
        self._end_effector.initialize(physics_sim_view)
        dof_names = self.dof_names if self.dof_names is not None else self.get_all_joint_names()
        gripper_default_state = np.array(self._default_joint_positions[-self.num_gripper_joints:], dtype=np.float64)
        self._gripper.set_default_state(gripper_default_state)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=dof_names,
        )
        self.set_joint_positions(self._default_joint_positions)

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

    # ── Gripper control methods ─────────────────────────────────────────────

    def open_gripper(self) -> None:
        """Open the gripper to the fully open position."""
        self._gripper.open()

    def close_gripper(self) -> None:
        """Close the gripper to the fully closed position."""
        self._gripper.close()
