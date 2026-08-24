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
import os

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleArticulation

from robots.base_robot import BaseRobot


class RMPFlowController(mg.MotionPolicyController):
    """RMPFlow motion controller

    Args:
        name (str): Controller name
        robot_articulation (SingleArticulation): Robot articulation object
        physics_dt (float, optional): Physics time step. Defaults to 1.0/60.0.
        use_default_config (bool, optional): Whether to use default config files. Defaults to True.
    """

    # Class-wide flag set by main.py BEFORE controllers are constructed (used by
    # position-only collection, cfg.collect_position_only). When True, RmpFlow
    # rolls out an internal "virtual robot" instead of reading the measured joint
    # state each frame, so the commanded position targets advance at full planned
    # speed even though the real arm (pure position PD, no velocity feed-forward)
    # tracks them with some lag — the exact same control situation as replay and
    # inference. Without this, position-only control makes RmpFlow's closed loop
    # converge to a crawl (targets stay glued to the lagging measured state).
    ignore_robot_state_updates: bool = False

    def __init__(
        self, 
        name: str, 
        robot_articulation: SingleArticulation, 
        physics_dt: float = 1.0 / 60.0,
        use_default_config: bool = True
    ) -> None:
        # The arm supplies its own Lula description. This is what makes the shared
        # controllers arm-agnostic: previously every one of them planned against
        # Franka's files no matter which robot was passed in, so any other arm either
        # refused to solve or drove to Franka's joint frames.
        robot_motion_config = None
        if isinstance(robot_articulation, BaseRobot):
            try:
                robot_motion_config = robot_articulation.motion_config
            except NotImplementedError:
                robot_motion_config = None

        if robot_motion_config is not None:
            self.rmp_flow_config = dict(robot_motion_config)
            self.rmp_flow_config.setdefault("maximum_substep_size", 0.00334)
            self.rmp_flow_config.setdefault("ignore_robot_state_updates", False)
        elif use_default_config:
            # Use system default RMPflow configuration
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
        else:
            # Use custom configuration file paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            rmpflow_dir = os.path.join(current_dir, "rmpflow")
            
            self.rmp_flow_config = {
                'end_effector_frame_name': 'right_gripper',
                'maximum_substep_size': 0.00334,
                'ignore_robot_state_updates': False,
                'robot_description_path': os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
                'urdf_path': os.path.join(current_dir, "lula_franka_gen.urdf"),
                'rmpflow_config_path': os.path.join(rmpflow_dir, "franka_rmpflow_common.yaml")
            }
        
        print(self.rmp_flow_config)
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)
        if RMPFlowController.ignore_robot_state_updates:
            # Virtual-robot rollout (see class flag above). After reset() the
            # internal state is None, so the first compute re-seeds it from the
            # measured robot state — no extra re-sync needed per episode.
            self.rmp_flow.set_ignore_state_updates(True)

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        self._default_position, self._default_orientation = self._kinematic_base_pose(robot_articulation)
        print(
            f"[rmpflow] base_pose={self._default_position} quat={self._default_orientation} "
            f"ee_frame={self.rmp_flow_config.get('end_effector_frame_name')} "
            f"urdf={os.path.basename(str(self.rmp_flow_config.get('urdf_path')))} "
            f"articulation_prim_pose={robot_articulation.get_world_pose()[0]}",
            flush=True,
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    @staticmethod
    def _kinematic_base_pose(robot_articulation):
        """World pose of the arm's base, read AFTER physics initialisation.

        Deliberately re-read in reset() rather than cached in __init__: controllers are
        built before the articulation is initialised, and at that point get_world_pose()
        still reports the authored prim transform. For an arm whose USD welds its base
        somewhere other than the prim origin (piper sits at [-0.20, -0.15, 0.80] no
        matter what the config asks for), the two differ and RMPFlow would plan in a
        frame shifted by that much.
        """
        return robot_articulation.get_world_pose()

    def reset(self):
        mg.MotionPolicyController.reset(self)
        # Re-read rather than reuse the pose captured in __init__. Controllers are
        # built before the robot's world pose is applied, so at construction time an
        # arm whose base_link is offset inside its USD (piper) reports the raw offset
        # [-0.20, -0.15, 0.80] instead of its placed position -- RMPFlow then plans in
        # a frame shifted by that much and the gripper stalls short of the target.
        self._default_position, self._default_orientation = self._kinematic_base_pose(
            self._articulation_motion_policy._robot_articulation
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
