# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleArticulation


class RMPFlowController(mg.MotionPolicyController):
    """RMPFlow motion controller for Piper robot.

    Args:
        name (str): Controller name.
        robot_articulation (SingleArticulation): Robot articulation object.
        physics_dt (float, optional): Physics time step. Defaults to 1.0/60.0.
        use_default_config (bool, optional): Whether to use default config files. Defaults to False.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 60.0,
        use_default_config: bool = False,
    ) -> None:
        # Piper must use custom config (no built-in Isaac Sim support)
        use_default_config = False

        # Custom configuration file paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rmpflow_dir = os.path.join(current_dir, "rmpflow")

        self.rmp_flow_config = {
            'end_effector_frame_name': 'link6',
            'maximum_substep_size': 0.00334,
            'ignore_robot_state_updates': False,
            'robot_description_path': os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
            'urdf_path': os.path.join(current_dir, "piper.urdf"),
            'rmpflow_config_path': os.path.join(rmpflow_dir, "piper_rmpflow_common.yaml")
        }

        print(self.rmp_flow_config)
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
