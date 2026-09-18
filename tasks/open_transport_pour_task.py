from typing import Any, Dict, Optional

import numpy as np
import omni.physx
from loguru import logger
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

from utils.franka_mimic_force import validate_franka_finger_speed, validate_single_mimic_finger_drive
from utils.gripper_contact import finger_pad_snapshot

from .base_task import BaseTask
from .liquid_task_mixin import LiquidTaskMixin


class _OpenTransportPourScene(BaseTask):
    """Level-4 composite task: Open door → Transport beaker → Pour.

    Scene objects are placed at fixed randomised ranges defined here.
    Paths for the primary beaker and target platform come from
    ``cfg.task.obj_paths``.
    """

    # Hardcoded scene randomisation ranges (x_range, y_range, z_fixed)
    _SCENE_OBJECTS = [
        ("/World/beaker2",        (0.04, 0.05),  (0.32, 0.33),  0.86),
        ("/World/conical_bottle02", (0.11, 0.12), (-0.41, -0.40), 0.86),
        ("/World/beaker1",        (0.17, 0.18),  (-0.155, -0.145), 0.86),
        ("/World/target_plat2",   (0.10, 0.11),  (-0.52, -0.51), 0.775),
        ("/World/target_plat",    (0.03, 0.04),  (0.54, 0.55),  0.775),
        ("/World/MuffleFurnace",  (0.69, 0.70),  (0.09, 0.10),  0.78),
    ]

    def __init__(self, cfg: Any, world: Any, stage: Any, robot: Any) -> None:
        super().__init__(cfg, world, stage, robot)
        self.beaker_path      = cfg.task.obj_paths[0]["path"]
        self.target_plat_path = cfg.task.obj_paths[1]["path"]
        self._placement_supports = None

    _GRASP_OBJECTS = ("/World/beaker2", "/World/conical_bottle02")

    def setup_objects(self) -> None:
        super().setup_objects()
        parameters = finger_pad_snapshot(self.cfg)
        if not parameters:
            return
        # Author physical contact only, before simulation initialization. The
        # rubber pads' render material and the shared robot USD are untouched.
        material = UsdShade.Material.Define(self.stage, self.robot.prim_path + "/ConicalPadContact")
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(parameters["static_friction"])
        physics.CreateDynamicFrictionAttr(parameters["dynamic_friction"])
        physics.CreateRestitutionAttr(0.0)
        compliance = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        compliance.CreateCompliantContactAccelerationSpringAttr(False)
        compliance.CreateCompliantContactStiffnessAttr(parameters["stiffness"])
        compliance.CreateCompliantContactDampingAttr(parameters["damping"])
        count = 0
        for name in ("panda_leftfinger", "panda_rightfinger"):
            finger = self.stage.GetPrimAtPath(self.robot.prim_path + "/" + name)
            for _ in range(3):
                instances = [prim for prim in Usd.PrimRange(finger) if prim.IsInstance()]
                if not instances:
                    break
                for prim in instances:
                    prim.SetInstanceable(False)
            for prim in Usd.PrimRange(finger):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy]).ComputeRelativeBound(prim, finger).ComputeAlignedRange()
                    logger.info("[conical-pad] collider={} bounds relative to finger: {} .. {} m",
                                prim.GetPath(), tuple(bounds.GetMin()), tuple(bounds.GetMax()))
                    contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    contact.CreateTorsionalPatchRadiusAttr(0.008)
                    contact.CreateMinTorsionalPatchRadiusAttr(parameters["min_torsional_radius"])
                    contact.CreateRestOffsetAttr(0.0)
                    contact.CreateContactOffsetAttr(parameters["contact_offset"])
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
                    count += 1
        if not count:
            raise ValueError("No physical Franka finger colliders found")
        logger.info("[conical-pad] {} colliders; physical contact {}", count, parameters)

    def _configure_gripper_drive(self) -> None:
        """Force-cap the existing single driven/mimic hand; keep position replay.

        Franka Hand specifies 30–70 N continuous grasp force. Only the lead
        finger is driven; independently powering the mimic would double-count.
        """
        force = getattr(self.cfg.task, "gripper_drive_force", None)
        if force is None:
            return
        force = float(force)
        if not 1 <= force <= 70:
            raise ValueError("Franka grasp force must be within its 70 N continuous limit")
        speed = getattr(self.cfg.task, "gripper_max_joint_velocity", None)
        if speed is not None:
            speed = float(speed)
            validate_franka_finger_speed(speed)
        validate_single_mimic_finger_drive(self.stage, self.robot.prim_path, allow_zero_gain_follower=True)
        joint = self.stage.GetPrimAtPath(self.robot.prim_path + "/panda_hand/panda_finger_joint1")
        drive = UsdPhysics.DriveAPI.Get(joint, "linear")
        if not drive:
            raise ValueError("Expected Franka single lead-finger linear drive")
        drive.CreateMaxForceAttr(force)
        drive.CreateStiffnessAttr(10000.0)
        drive.CreateDampingAttr(200.0)
        indices = np.array([self.robot.gripper.joint_dof_indicies[0]], dtype=int)
        self.robot.get_articulation_controller().set_max_efforts(np.array([force]), joint_indices=indices)
        self.robot._articulation_view.set_gains(kps=np.array([[10000.0]]), kds=np.array([[200.0]]), joint_indices=indices)
        if speed is not None:
            # Both fingers need the same physical speed envelope. The asset's
            # passive mimic allows 10 m/s; limiting only the driven finger
            # leaves contact/mimic impulses free to accelerate its follower.
            for number in (1, 2):
                finger_joint = self.stage.GetPrimAtPath(
                    self.robot.prim_path + f"/panda_hand/panda_finger_joint{number}"
                )
                PhysxSchema.PhysxJointAPI.Apply(finger_joint).CreateMaxJointVelocityAttr(speed)
            finger_indices = np.asarray(self.robot.gripper.joint_dof_indicies, dtype=int)
            self.robot._articulation_view.set_max_joint_velocities(
                np.full((1, len(finger_indices)), speed), joint_indices=finger_indices
            )
            logger.info("[conical-gripper] both physical finger velocity limits = {} m/s", speed)
        api = PhysxSchema.PhysxArticulationAPI.Apply(self.stage.GetPrimAtPath(self.robot.prim_path))
        api.CreateSolverPositionIterationCountAttr(64)
        api.CreateSolverVelocityIterationCountAttr(16)
        logger.info("[conical-gripper] lead drive capped at {} N; position command, passive mimic unchanged", force)

    def _apply_grasp_friction(self) -> None:
        """Raise gripper<->object contact friction on the grasped glassware so the
        near-zero-squeeze grasp holds through transport/pour instead of slipping,
        and reproduces in open-loop replay. Realistic glass-on-gripper µ — a
        scene-physics correction. Applied in BOTH collect and replay reset for
        parity. No-op unless task.grasp_friction is set."""
        self._configure_gripper_drive()
        task = getattr(self.cfg, "task", None)
        mu = getattr(task, "grasp_friction", None) if task else None
        if mu is None:
            return
        for path in self._GRASP_OBJECTS:
            self.object_utils.set_physics_friction(
                path, static_friction=float(mu), dynamic_friction=float(mu)
            )

    def reset(self) -> None:
        super().reset()
        self._placement_supports = None
        self.robot.initialize()
        for obj_path, x_range, y_range, z in self._SCENE_OBJECTS:
            pos = np.array([np.random.uniform(*x_range), np.random.uniform(*y_range), z])
            self.object_utils.set_object_position(obj_path, pos)
            self._record_object_pose(obj_path)
        self._apply_grasp_friction()

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)
        self._placement_supports = init_state.get("extra", {}).get("placement_supports")
        self._apply_grasp_friction()

    def _measure_placement_supports(self) -> dict:
        """The authored target sheets are non-colliding markers above the desk.

        Before the first robot action, raycast beneath each empty target area to
        find its actual physical support, rather than dropping onto marker Z.
        """
        supports = {}
        query = omni.physx.get_physx_scene_query_interface()
        for path in ("/World/target_plat", "/World/target_plat2"):
            center = np.asarray(self.object_utils.get_geometry_center(object_path=path), dtype=float)
            origin = center + [0, 0, 0.015]
            hit = query.raycast_closest(tuple(origin), (0.0, 0.0, -1.0), 0.3)
            if not hit.get("hit"):
                raise ValueError(f"No physical support below placement marker: {path}")
            point = np.asarray(hit["position"], dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError(f"Invalid placement support raycast: {path}")
            supports[path] = point.tolist()
            logger.info("[conical-support] {} marker_z={:.4f} physical_z={:.4f} collider={}",
                        path, center[2], point[2], hit.get("collision", ""))
        return supports

    def step(self) -> Optional[Dict[str, Any]]:
        self.frame_idx += 1
        if not self.check_frame_limits(max_steps=self.cfg.task.max_steps):
            return None
        state = self.get_basic_state_info(
            object_path=self.beaker_path,
            target_path=self.target_plat_path,
        )
        if state is None:
            return None
        if getattr(self.cfg, "liquid", None) is not None and self.cfg.liquid.enabled:
            if self._placement_supports is None:
                self._placement_supports = self._measure_placement_supports()
                self._episode_init_state["extra"]["placement_supports"] = self._placement_supports
            state["placement_supports"] = self._placement_supports
        return state


class OpenTransportPourTask(LiquidTaskMixin, _OpenTransportPourScene):
    LIQUID_REPLAY_SUPPORTS = ("/World/target_plat", "/World/target_plat2", "/World/MuffleFurnace")
    LIQUID_REPLAY_ARTICULATIONS = ("/World/MuffleFurnace",)
