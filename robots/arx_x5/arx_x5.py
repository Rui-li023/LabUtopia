# SPDX-License-Identifier: Apache-2.0

"""ARX X5 - 6-DOF arm with a parallel gripper.

Its URDF already ships a ``gripper_center`` link between the pads, so unlike Piper no
tool-centre frame has to be synthesised. Like Piper its USD welds the base, so the
configured position is re-applied in post_reset().
"""

import os

import carb
import numpy as np
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper

from robots.base_robot import BaseRobot
from utils.object_utils import ObjectUtils


class ArxX5(BaseRobot):
    """ARX X5 robotic arm.

    Args:
        prim_path: USD prim path for the robot.
        name: Robot name.
        usd_path: Optional override for the robot USD.
        position: Base position.
        orientation: Base orientation.
        default_joint_positions: Optional home configuration.
    """

    # 6 arm joints + 2 gripper fingers, home = the IK solution for the centre of the
    # intended workspace. Mid-range defaults are wrong for this arm: joint2 and joint3
    # are both one-sided ([0, 3.66] and [0, 3.14]), so their midpoint is a folded,
    # arm-up pose. RMPFlow's nullspace attractor then holds the gripper ~0.36 m above
    # the beaker and it never descends. This home has 0.39 rad of limit margin.
    DEFAULT_JOINT_POSITIONS = np.array([0.0021, 1.4444, 1.0570, -1.1823, 0.0, 0.0021, 0.044, 0.044])

    _ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    # joint8 is a mimic follower of gripper_joint in the URDF, but PhysX exposes both,
    # so both are commanded with the same sign.
    _GRIPPER_JOINT_NAMES = ["gripper_joint", "joint8"]

    # Prismatic fingers, 0 (closed) .. 0.044 (open) each -> 88 mm total opening, wider
    # than the Franka's 80 mm.
    GRIPPER_OPEN_M = 0.044

    def __init__(
        self,
        prim_path: str = "/World/ArxX5",
        name: str = "arx_x5",
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
                usd_path = os.path.join(repo_root, "assets/robots/arx_x5.usd")
            if not os.path.exists(usd_path):
                carb.log_error(f"Could not find ARX X5 USD at {usd_path}")
                raise FileNotFoundError(f"ARX X5 USD not found: {usd_path}")
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        self._end_effector_prim_path = prim_path + "/link6"

        super().__init__(prim_path=prim_path, name=name, position=position, orientation=orientation)

        self._gripper_open_position = np.array([self.GRIPPER_OPEN_M, self.GRIPPER_OPEN_M])
        self._gripper_closed_position = np.array([0.0, 0.0])

        self._gripper = ParallelGripper(
            end_effector_prim_path=self._end_effector_prim_path,
            joint_prim_names=self._GRIPPER_JOINT_NAMES.copy(),
            joint_opened_positions=self._gripper_open_position,
            joint_closed_positions=self._gripper_closed_position,
            action_deltas=np.array([0.01, 0.01]) / get_stage_units(),
        )

    # ── BaseRobot contract ───────────────────────────────────────────────────

    @property
    def arm_joint_names(self) -> list[str]:
        return self._ARM_JOINT_NAMES

    @property
    def gripper_joint_names(self) -> list[str]:
        return self._GRIPPER_JOINT_NAMES

    @property
    def gripper_distance_multipliers(self) -> list[float]:
        """Both fingers open in the positive direction (prismatic 0..0.044)."""
        return [1.0, 1.0]

    @property
    def end_effector_prim_path(self) -> str:
        return self._end_effector_prim_path

    @property
    def gripper_center_prim_path(self) -> str:
        """The URDF's own frame between the pads -- no synthesised offset needed."""
        return self.prim_path_str + "/gripper_center"

    @property
    def tool_frame_correction_euler_deg(self) -> list[float]:
        """Approach is local +X here, not +Z (measured by FK); fingers stay on +Y.

        Ry(-90) maps local +X onto the canonical +Z and leaves +Y untouched. Ry(+90)
        sends +X to -Z instead, which points the gripper away from the object and makes
        every grasp target unsolvable.
        """
        return [0.0, -90.0, 0.0]

    @property
    def ik_end_effector_frame(self) -> str:
        return "gripper_center"

    @property
    def motion_config(self) -> dict:
        """Lula files generated by scripts/urdf_to_usd/make_lula_descriptor.py."""
        package_dir = os.path.dirname(os.path.abspath(__file__))
        rmpflow_dir = os.path.join(package_dir, "rmpflow")
        return {
            "end_effector_frame_name": "gripper_center",
            "maximum_substep_size": 0.00334,
            "robot_description_path": os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
            "urdf_path": os.path.join(package_dir, "arx_x5.urdf"),
            "rmpflow_config_path": os.path.join(rmpflow_dir, "arx_x5_rmpflow_common.yaml"),
        }

    def get_gripper_position(self) -> np.ndarray:
        return ObjectUtils.get_instance().get_object_xform_position(object_path=self.gripper_center_prim_path)

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view)
        self._end_effector = SingleRigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)

        dof_names = self.dof_names if self.dof_names is not None else self.get_all_joint_names()
        self._gripper.set_default_state(
            np.array(self._default_joint_positions[-self.num_gripper_joints :], dtype=np.float64)
        )
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
        for dof_index in self._gripper.joint_dof_indicies:
            self._articulation_controller.switch_dof_control_mode(dof_index=dof_index, mode="position")
        self.set_joint_positions(self._default_joint_positions)
