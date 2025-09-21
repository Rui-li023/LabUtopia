# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import List, Optional

import carb
import numpy as np
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.sensor import ContactSensor
from omni.isaac.sensor import Camera
from utils.object_utils import ObjectUtils
class Franka(Robot):
    """[summary]

    Args:
        prim_path (str): [description]
        name (str, optional): [description]. Defaults to "franka_robot".
        usd_path (Optional[str], optional): [description]. Defaults to None.
        position (Optional[np.ndarray], optional): [description]. Defaults to None.
        orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
        gripper_dof_names (Optional[List[str]], optional): [description]. Defaults to None.
        gripper_open_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        gripper_closed_position (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

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
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        self.prim_path_str = prim_path
        
        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
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
                
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        
        if gripper_dof_names is not None:
            if deltas is None:
                deltas = np.array([0.05, 0.05]) / get_stage_units()
            self._gripper = ParallelGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=gripper_dof_names,
                joint_opened_positions=gripper_open_position,
                joint_closed_positions=gripper_closed_position,
                action_deltas=deltas,
            )
            self._gripper.set_action_deltas(np.array([0.03, 0.03]) / get_stage_units())
        
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
        self.camera = Camera(
            prim_path=prim_path + "/panda_hand/arm_camera",
            translation=np.array([-0.2, -0, -0.02]),
            frequency=60,
            resolution=(256, 256),
            orientation=np.array([0.20083, 0.67799, -0.67799, -0.20083])
        )
        self.camera.set_local_pose(orientation=np.array([0.20083, 0.67799, -0.67799, -0.20083]), camera_axes="usd")
        self.camera.set_clipping_range(near_distance=0.05)
        self.camera.set_focal_length(1.)
        return

    def get_contact_sensor(self):
        
        return self.left_contact_sensor, self.right_contact_sensor
        
    @property
    def end_effector(self) -> RigidPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> ParallelGripper:
        """[summary]

        Returns:
            ParallelGripper: [description]
        """
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        return

    def post_reset(self) -> None:
        """[summary]"""
        super().post_reset()
        self._gripper.post_reset()
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[1], mode="position"
        )
        return

    def get_gripper_position(self) -> np.ndarray:
        """[summary]

        Returns:
            np.ndarray: [description]
        """
        return ObjectUtils.get_instance().get_object_xform_position(
                object_path=self.prim_path_str + "/panda_hand/tool_center"
            )