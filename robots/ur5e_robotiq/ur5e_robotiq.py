# SPDX-License-Identifier: Apache-2.0

"""Universal Robots UR5e with a Robotiq 2F-85 gripper.

The UR arms ship with a bare tool flange, so the pick task needs a gripper bolted on.
This is one articulation built from a combined xacro (ur5e_macro + Robotiq85), not two
assets referenced together: PhysX cannot drive a gripper that lives in its own
articulation.

Unlike every other arm here the Robotiq is REVOLUTE -- one driver joint from 0 rad
(open) to 0.81 rad (closed), with five mimic followers. The shared controllers speak in
finger separation (metres), so the width-to-angle mapping is overridden below; taking
the default prismatic mapping would command 0.028 rad for a 28 mm grasp, which is
essentially still open.
"""

import os

import carb
import numpy as np
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper

from robots.base_robot import BaseRobot
from utils.object_utils import ObjectUtils


class UR5eRobotiq(BaseRobot):
    """UR5e + Robotiq 2F-85."""

    # Subclasses can keep the shared Robotiq control semantics while selecting the
    # link and joints authored by their vendor-specific combined URDF.
    GRIPPER_BASE_LINK_NAME = "gripper_base"

    # The articulation contains the gripper driver plus five mimic followers. Only
    # the driver is an actuator; PhysX carries the followers through mimic constraints.
    DEFAULT_JOINT_POSITIONS = np.array([0.0, -1.2, 1.4, -1.75, -1.57, 0.0] + [0.0] * 6)

    _ARM_JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    _GRIPPER_JOINT_NAMES = ["gripper_joint"]
    _MIMIC_JOINT_NAMES = (
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "left_inner_finger_joint",
        "right_inner_finger_joint",
    )
    MIMIC_NATURAL_FREQUENCY = 50.0
    MIMIC_DAMPING_RATIO = 1.0

    # 2F-85 datasheet: 85 mm stroke, closing over 0..0.81 rad of the driver.
    GRIPPER_MAX_WIDTH_M = 0.085
    GRIPPER_CLOSED_RAD = 0.81

    def __init__(
        self,
        prim_path: str = "/World/UR5eRobotiq",
        name: str = "ur5e_robotiq",
        usd_path: str | None = None,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        default_joint_positions: np.ndarray | None = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._default_joint_positions = (
            default_joint_positions if default_joint_positions is not None else self.DEFAULT_JOINT_POSITIONS.copy()
        )

        if not prim.IsValid():
            if usd_path is None:
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                usd_path = os.path.join(repo_root, "assets/robots/ur5e_robotiq85.usd")
            if not os.path.exists(usd_path):
                carb.log_error(f"Could not find UR5e+Robotiq USD at {usd_path}")
                raise FileNotFoundError(f"UR5e+Robotiq USD not found: {usd_path}")
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        self._tune_mimic_constraints(prim_path)

        self._end_effector_prim_path = prim_path + "/" + self.GRIPPER_BASE_LINK_NAME

        super().__init__(prim_path=prim_path, name=name, position=position, orientation=orientation)

        self._gripper_open_position = np.array([0.0])
        self._gripper_closed_position = np.array([self.GRIPPER_CLOSED_RAD])

        self._gripper = ParallelGripper(
            end_effector_prim_path=self._end_effector_prim_path,
            joint_prim_names=self._GRIPPER_JOINT_NAMES.copy(),
            joint_opened_positions=self._gripper_open_position,
            joint_closed_positions=self._gripper_closed_position,
            # Robotiq closes as its revolute driver angle INCREASES. ParallelGripper's
            # delta mode implements close as current-delta, which silently reversed
            # this mechanism (close stayed at the 0-rad lower limit while open moved
            # toward 0.81). Absolute targets preserve the URDF convention.
            action_deltas=None,
            use_mimic_joints=True,
        )

    @classmethod
    def _tune_mimic_constraints(cls, prim_path: str) -> None:
        """Make the imported four-bar constraints track without long oscillation."""
        for joint_name in cls._MIMIC_JOINT_NAMES:
            joint = get_prim_at_path(f"{prim_path}/joints/{joint_name}")
            if not joint.IsValid():
                raise RuntimeError(f"Missing Robotiq mimic joint: {joint.GetPath()}")
            damping = joint.GetAttribute("physxMimicJoint:rotX:dampingRatio")
            frequency = joint.GetAttribute("physxMimicJoint:rotX:naturalFrequency")
            if not damping.IsValid() or not frequency.IsValid():
                raise RuntimeError(f"Robotiq joint {joint.GetPath()} has no rotX mimic constraint")
            damping.Set(cls.MIMIC_DAMPING_RATIO)
            frequency.Set(cls.MIMIC_NATURAL_FREQUENCY)

    # ── BaseRobot contract ───────────────────────────────────────────────────

    @property
    def arm_joint_names(self) -> list[str]:
        return self._ARM_JOINT_NAMES

    @property
    def gripper_joint_names(self) -> list[str]:
        return self._GRIPPER_JOINT_NAMES

    @property
    def end_effector_prim_path(self) -> str:
        return self._end_effector_prim_path

    @property
    def gripper_center_prim_path(self) -> str:
        return self.prim_path_str + "/tool_frame"

    @property
    def ik_end_effector_frame(self) -> str:
        return "tool_frame"

    @property
    def motion_config(self) -> dict:
        package_dir = os.path.dirname(os.path.abspath(__file__))
        rmpflow_dir = os.path.join(package_dir, "rmpflow")
        return {
            "end_effector_frame_name": "tool_frame",
            "maximum_substep_size": 0.00334,
            "robot_description_path": os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
            "urdf_path": os.path.join(package_dir, "ur5e_robotiq85.urdf"),
            "rmpflow_config_path": os.path.join(rmpflow_dir, "ur5e_robotiq_rmpflow_common.yaml"),
        }

    # ── Revolute-gripper overrides ───────────────────────────────────────────

    @property
    def default_pick_gripper_distance(self) -> float:
        """Fully open, expressed as a finger separation like every other arm."""
        return self.GRIPPER_MAX_WIDTH_M

    def get_gripper_joint_targets_from_distance(self, distance: float) -> np.ndarray:
        """Map a finger separation in metres to the 2F-85's driver angle.

        Linear over the stroke: 0 rad at 85 mm, 0.81 rad closed. The base class assumes
        prismatic fingers where the distance IS the joint value, which for this gripper
        would leave it wide open at any sensible grasp width.
        """
        width = float(np.clip(distance, 0.0, self.GRIPPER_MAX_WIDTH_M))
        angle = (1.0 - width / self.GRIPPER_MAX_WIDTH_M) * self.GRIPPER_CLOSED_RAD
        return np.array([angle], dtype=np.float64)

    def get_gripper_position(self) -> np.ndarray:
        return ObjectUtils.get_instance().get_object_xform_position(object_path=self.gripper_center_prim_path)

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view)
        self._end_effector = SingleRigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)

        dof_names = self.dof_names if self.dof_names is not None else self.get_all_joint_names()
        # Explicit, not a tail slice of the default vector: the driver is followed by
        # five mimic DOFs in the articulation.
        self._gripper.set_default_state(self._gripper_open_position)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=dof_names,
        )
        self.enforce_requested_world_pose()
        self.set_joint_positions(self._default_joint_positions)

    def post_reset(self) -> None:
        super().post_reset()
        self._gripper.post_reset()
        for dof_index in self._gripper.active_joint_indices:
            self._articulation_controller.switch_dof_control_mode(dof_index=dof_index, mode="position")
        self.set_joint_positions(self._default_joint_positions)
