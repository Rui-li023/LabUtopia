# SPDX-License-Identifier: Apache-2.0

"""Split/Mobile ALOHA controlled through its front-left arm."""

import os

import carb
import numpy as np
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, get_stage_units
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from pxr import Gf, UsdGeom

from robots.base_robot import BaseRobot
from utils.object_utils import ObjectUtils


class SplitAlohaFrontLeft(BaseRobot):
    """Full Split ALOHA articulation with only the front-left arm exposed."""

    # fl_joint7/8 start 73.574 mm from link6, while the actual inner pad faces
    # are another 58.430 mm along the fingers.  Using only the joint origin puts
    # the reported TCP at the slide roots, so the visible jaws stop 5.8 cm past
    # every commanded grasp point and close on empty space.
    TOOL_CENTER_OFFSET_M = 0.073574 + 0.058430
    GRIPPER_OPEN_M = 0.044

    _ARM_JOINT_NAMES = [f"fl_joint{index}" for index in range(1, 7)]
    _GRIPPER_JOINT_NAMES = ["fl_joint7", "fl_joint8"]

    # The importer orders this branched articulation by graph depth, interleaving the
    # four arms. Keep all 42 entries explicit so the three parked arms and ten wheel
    # joints receive deterministic position targets on every reset.
    DEFAULT_JOINT_POSITIONS = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # castors and drive wheels at graph depth 1
            0.0021,
            0.0021,
            0.0021,
            0.0021,  # fl/fr/lr/rr joint1
            0.0,
            0.0,
            0.0,
            0.0,  # remaining castor wheels
            1.4444,
            1.4444,
            1.4444,
            1.4444,  # joint2
            1.0570,
            1.0570,
            1.0570,
            1.0570,  # joint3
            -1.1823,
            -1.1823,
            -1.1823,
            -1.1823,  # joint4
            0.0,
            0.0,
            0.0,
            0.0,  # joint5
            0.0021,
            0.0021,
            0.0021,
            0.0021,  # joint6
            GRIPPER_OPEN_M,
            GRIPPER_OPEN_M,
            GRIPPER_OPEN_M,
            GRIPPER_OPEN_M,
            GRIPPER_OPEN_M,
            GRIPPER_OPEN_M,
            GRIPPER_OPEN_M,
            GRIPPER_OPEN_M,  # fl/fr/lr/rr finger pairs
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        prim_path: str = "/World/SplitAloha",
        name: str = "split_aloha_fl",
        usd_path: str | None = None,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        default_joint_positions: np.ndarray | None = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._default_joint_positions = np.asarray(
            default_joint_positions if default_joint_positions is not None else self.DEFAULT_JOINT_POSITIONS.copy(),
            dtype=np.float64,
        )

        if not prim.IsValid():
            if usd_path is None:
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                usd_path = os.path.join(repo_root, "assets/robots/split_aloha.usd")
            if not os.path.exists(usd_path):
                carb.log_error(f"Could not find Split ALOHA USD at {usd_path}")
                raise FileNotFoundError(f"Split ALOHA USD not found: {usd_path}")
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        self._end_effector_prim_path = prim_path + "/fl_link6"
        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
        )

        self._gripper_open_position = np.full(2, self.GRIPPER_OPEN_M, dtype=np.float64)
        self._gripper_closed_position = np.zeros(2, dtype=np.float64)
        self._gripper = ParallelGripper(
            end_effector_prim_path=self._end_effector_prim_path,
            joint_prim_names=self._GRIPPER_JOINT_NAMES.copy(),
            joint_opened_positions=self._gripper_open_position,
            joint_closed_positions=self._gripper_closed_position,
            action_deltas=np.full(2, 0.01, dtype=np.float64) / get_stage_units(),
        )

        # The vendor USD has no TCP prim. This runtime Xform matches the fixed frame
        # in split_aloha_fl.urdf and the physical gripping-pad centre.
        tool_center = UsdGeom.Xform.Define(get_current_stage(), self._end_effector_prim_path + "/fl_gripper_center")
        tool_center.AddTranslateOp().Set(Gf.Vec3d(self.TOOL_CENTER_OFFSET_M, 0.0, 0.0))

    @property
    def arm_joint_names(self) -> list[str]:
        return self._ARM_JOINT_NAMES

    @property
    def gripper_joint_names(self) -> list[str]:
        return self._GRIPPER_JOINT_NAMES

    @property
    def gripper_distance_multipliers(self) -> list[float]:
        return [1.0, 1.0]

    @property
    def default_pick_gripper_distance(self) -> float:
        return self.GRIPPER_OPEN_M

    @property
    def end_effector_prim_path(self) -> str:
        return self._end_effector_prim_path

    @property
    def gripper_center_prim_path(self) -> str:
        return self._end_effector_prim_path + "/fl_gripper_center"

    @property
    def tool_frame_correction_euler_deg(self) -> list[float]:
        # The ARX wrist approaches along local +X while the canonical frame uses +Z.
        return [0.0, -90.0, 0.0]

    @property
    def ik_end_effector_frame(self) -> str:
        return "fl_gripper_center"

    @property
    def motion_config(self) -> dict:
        package_dir = os.path.dirname(os.path.abspath(__file__))
        rmpflow_dir = os.path.join(package_dir, "rmpflow")
        return {
            "end_effector_frame_name": "fl_gripper_center",
            "maximum_substep_size": 0.00334,
            "robot_description_path": os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
            "urdf_path": os.path.join(package_dir, "split_aloha_fl.urdf"),
            "rmpflow_config_path": os.path.join(rmpflow_dir, "split_aloha_fl_rmpflow_common.yaml"),
        }

    def get_gripper_position(self) -> np.ndarray:
        return ObjectUtils.get_instance().get_object_xform_position(object_path=self.gripper_center_prim_path)

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view)
        if self.num_dof != len(self._default_joint_positions):
            raise ValueError(
                f"Split ALOHA default state has {len(self._default_joint_positions)} values "
                f"for {self.num_dof} articulation DOFs"
            )
        self._end_effector = SingleRigidPrim(
            prim_path=self._end_effector_prim_path,
            name=self.name + "_end_effector",
        )
        self._end_effector.initialize(physics_sim_view)

        self._gripper.set_default_state(self._gripper_open_position)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        self.enforce_requested_world_pose()
        self.set_joints_default_state(
            positions=self._default_joint_positions,
            velocities=np.zeros(self.num_dof, dtype=np.float64),
        )
        self.set_joint_positions(self._default_joint_positions)
        self.set_joint_velocities(np.zeros(self.num_dof, dtype=np.float64))

    def post_reset(self) -> None:
        super().post_reset()
        self._gripper.post_reset()
        for dof_index in self.get_gripper_joint_indices():
            self._articulation_controller.switch_dof_control_mode(
                dof_index=dof_index,
                mode="position",
            )
        self.enforce_requested_world_pose()
        self.set_joint_positions(self._default_joint_positions)
        self.set_joint_velocities(np.zeros(self.num_dof, dtype=np.float64))
