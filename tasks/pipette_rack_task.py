import os
from typing import Any

import numpy as np
from loguru import logger
from omni.physx import get_physx_simulation_interface
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

from utils.franka_mimic_force import validate_single_mimic_finger_drive
from utils.gripper_contact import PipetteContactParameters
from utils.pipette_rack_material import configure_acrylic_rack_contact

from .base_task import BaseTask


class PipetteRackTask(BaseTask):
    """Pick pipette from table and place onto pipette rack.

    ``obj_paths[0]`` (pipette) may have complex USD transforms requiring
    geometry correction.  ``obj_paths[1]`` (rack) uses standard positioning.
    """

    def setup_objects(self) -> None:
        rack_contact = self.cfg.task.get("rack_contact")
        if rack_contact is not None and self.cfg.task.obj_paths[1].get("friction") is not None:
            raise ValueError(
                "rack_contact cannot be combined with whole-rack friction (silicone feet must be preserved)"
            )
        gripper_cfg = getattr(getattr(self.cfg, "robot", None), "gripper", None)
        single_mimic_force = getattr(gripper_cfg, "force_semantics", "legacy") == "single_mimic_drive"
        if getattr(self.cfg.task, "force_gripper_servo", False) or single_mimic_force:
            contact_parameters = PipetteContactParameters.from_config(self.cfg)
            if single_mimic_force:
                # Read-only and before contact authoring. Never add a second
                # actuator to obtain the requested force from both fingers.
                topology = validate_single_mimic_finger_drive(self.stage, self.robot.prim_path_str)
                logger.info("Validated single-mimic gripper force topology: {}", topology)
            self.robot.enable_gripper_force_servo(
                speed=contact_parameters.closing_speed,
                opening_speed=contact_parameters.opening_speed,
                position_iterations=int(getattr(self.cfg.task, "gripper_solver_position_iterations", 64)),
                velocity_iterations=int(getattr(self.cfg.task, "gripper_solver_velocity_iterations", 16)),
                drive_mode=str(getattr(self.cfg.task, "gripper_force_servo_mode", "velocity")),
                opening_width=float(getattr(self.cfg.task, "gripper_opening_width", 0.08)),
            )
            radius = float(getattr(self.cfg.task, "gripper_pad_min_torsional_radius", 0.0))
            if not np.isfinite(radius) or not 0 <= radius <= 0.004:
                raise ValueError("Finger contact patch must be finite and in [0, 4] mm")
            configure_friction = bool(getattr(self.cfg.task, "gripper_pad_friction", False))
            if (
                radius > 0
                or configure_friction
                or contact_parameters.contact_stiffness > 0
                or contact_parameters.contact_offset > 0
            ):
                count = 0
                pad_material = None
                if configure_friction or contact_parameters.contact_stiffness > 0:
                    # Explicit dry pad/ABS engineering estimates, not a
                    # measured material calibration or a sticking constraint.
                    pad_material = UsdShade.Material.Define(self.stage, self.robot.prim_path_str + "/PadContact")
                    material_api = UsdPhysics.MaterialAPI.Apply(pad_material.GetPrim())
                    material_api.CreateStaticFrictionAttr(contact_parameters.static_friction)
                    material_api.CreateDynamicFrictionAttr(contact_parameters.dynamic_friction)
                    material_api.CreateRestitutionAttr(0.0)
                    if contact_parameters.contact_stiffness > 0:
                        compliance = PhysxSchema.PhysxMaterialAPI.Apply(pad_material.GetPrim())
                        compliance.CreateCompliantContactAccelerationSpringAttr(False)
                        compliance.CreateCompliantContactStiffnessAttr(contact_parameters.contact_stiffness)
                        compliance.CreateCompliantContactDampingAttr(contact_parameters.contact_damping)
                for name in ("panda_leftfinger", "panda_rightfinger"):
                    finger = self.stage.GetPrimAtPath(self.robot.prim_path_str + "/" + name)
                    # Isaac's finger collision meshes are instanceable USD
                    # references. Author the contact metadata on this robot's
                    # local instance, never in the shared NVIDIA asset.
                    for _ in range(3):
                        instances = [prim for prim in Usd.PrimRange(finger) if prim.IsInstance()]
                        if not instances:
                            break
                        for prim in instances:
                            prim.SetInstanceable(False)
                    for prim in Usd.PrimRange(finger):
                        if prim.HasAPI(UsdPhysics.CollisionAPI):
                            contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                            if radius > 0:
                                contact.CreateTorsionalPatchRadiusAttr(0.008)
                                contact.CreateMinTorsionalPatchRadiusAttr(radius)
                            if contact_parameters.contact_offset > 0:
                                contact.CreateContactOffsetAttr(contact_parameters.contact_offset)
                                contact.CreateRestOffsetAttr(0.0)
                            if pad_material is not None:
                                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                                    pad_material, UsdShade.Tokens.strongerThanDescendants, "physics"
                                )
                            count += 1
                if not count:
                    raise RuntimeError("No physical finger colliders found for the contact-patch configuration")
                logger.info(
                    "Configured {} finger colliders with {:.1f} mm minimum torsional contact radius",
                    count,
                    radius * 1000,
                )
            logger.info("Pipette contact parameters: {}", contact_parameters)
        super().setup_objects()
        if rack_contact is not None:
            configure_acrylic_rack_contact(
                self.stage,
                str(self.cfg.task.obj_paths[1].path),
                static_friction=float(rack_contact.static_friction),
                dynamic_friction=float(rack_contact.dynamic_friction),
            )
        if os.environ.get("LABUTOPIA_PIPETTE_CONTACTS") in ("1", "grasp"):
            for obj_cfg in self.cfg.task.obj_paths:
                prim = self.stage.GetPrimAtPath(obj_cfg["path"])
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            self._contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(
                self._report_contacts
            )

    def _report_contacts(self, headers, contacts) -> None:
        for header in headers:
            paths = [str(PhysicsSchemaTools.intToSdfPath(value)) for value in (header.collider0, header.collider1)]
            selector = "finger" if os.environ.get("LABUTOPIA_PIPETTE_CONTACTS") == "grasp" else "pipette_rack"
            if not any(selector in path for path in paths):
                continue
            for contact in contacts[header.contact_data_offset : header.contact_data_offset + header.num_contact_data]:
                penetration = max(0.0, -float(contact.separation))
                if penetration > getattr(self, "_contact_peak_penetration", 0.0):
                    self._contact_peak_penetration = penetration
                    logger.debug("Pipette contact penetration peak: meters={:.8f}, pair={}", penetration, paths)
                if getattr(self, "frame_idx", 0) % 20:
                    continue
                local = None
                hand_local = None
                point = Gf.Vec3d(contact.position.x, contact.position.y, contact.position.z)
                hand = self.stage.GetPrimAtPath(self.robot.prim_path_str + "/panda_hand")
                if hand.IsValid():
                    hand_local = UsdGeom.XformCache().GetLocalToWorldTransform(hand).GetInverse().Transform(point)
                roots = [path.rsplit("/Body", 1)[0] for path in paths if "pipette" in path.lower() and "/Body" in path]
                if roots:
                    transform = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(roots[0]))
                    local = transform.GetInverse().Transform(point)
                logger.debug(
                    "Pipette CONTACT frame={} pair={} point={} normal={} impulse={} "
                    "body_local={} separation={} hand_local={}",
                    self.frame_idx,
                    paths,
                    contact.position,
                    contact.normal,
                    contact.impulse,
                    local,
                    contact.separation,
                    hand_local,
                )

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        # Reinitialization can restore authored position-drive gains. Match the
        # replay reset path so force/velocity grasps use the same actual drives.
        if getattr(self.robot, "_gripper_control_mode", "position") != "position":
            self.robot.post_reset()

        self.source_obj = self.cfg.task.obj_paths[0]["path"]  # pipette
        self.target_obj = self.cfg.task.obj_paths[1]["path"]  # rack

        # Pipette needs geometry correction due to pivot/rotation/scale
        self._randomize_with_geometry_correction(self.source_obj, self.cfg.task.obj_paths[0]["position_range"])
        # Rack has standard positioning
        self.randomize_object_position(self.target_obj, self.cfg.task.obj_paths[1]["position_range"])

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)
        self.source_obj = self.cfg.task.obj_paths[0]["path"]
        self.target_obj = self.cfg.task.obj_paths[1]["path"]

    def step(self) -> dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None
        info = {
            "target_seat_position": self._slot_position(),
            "object_grasp_position": self._grasp_position(),
        }
        if os.environ.get("LABUTOPIA_PIPETTE_GRASP_DEBUG") == "1":
            # Match the panda_hand frame used by Lula FK, not the Franka
            # wrapper's default end-effector prim (panda_rightfinger).
            hand_path = self.robot.prim_path_str + "/panda_hand"
            if self.stage.GetPrimAtPath(hand_path).IsValid():
                info["physical_hand_quaternion"] = self._world_quaternion(hand_path)
        if self.stage.GetPrimAtPath(self.source_obj + "/AxisHead").IsValid():
            info.update(
                object_axis=self._marker_position(self.source_obj + "/AxisHead")
                - self._marker_position(self.source_obj + "/AxisBase"),
                target_axis=self._marker_position(self.target_obj + "/AxisHead")
                - self._marker_position(self.target_obj + "/AxisBase"),
                object_quaternion=self._world_quaternion(self.source_obj),
                target_quaternion=self._world_quaternion(self.target_obj + "/PipetteSeat"),
                object_reference_position=self._reference_position(),
            )
        else:
            info.update(
                object_axis=self._pipette_axis(self.source_obj + "/mesh_003", self.source_obj + "/mesh_004"),
                target_axis=self._pipette_axis(
                    self.target_obj + "/group_000/mesh_000", self.target_obj + "/group_000/mesh_001"
                ),
            )
        return self.get_basic_state_info(
            object_path=self.source_obj,
            target_path=self.target_obj,
            additional_info=info,
        )

    def _marker_position(self, path: str) -> np.ndarray:
        return np.asarray(
            UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path)).ExtractTranslation()
        )

    def _reference_position(self) -> np.ndarray:
        marker = self.source_obj + "/Reference"
        if self.stage.GetPrimAtPath(marker).IsValid():
            return self._marker_position(marker)
        # Backward compatibility with the recorded 220 mm asset.
        return np.asarray(
            UsdGeom.XformCache()
            .GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.source_obj))
            .Transform(Gf.Vec3d(0, 0, 0.110))
        )

    def _world_quaternion(self, path: str) -> np.ndarray:
        quat = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path)).ExtractRotationQuat()
        return np.asarray([*quat.GetImaginary(), quat.GetReal()])

    def _record_object_pose(self, obj_path: str) -> None:
        # Gf uses row-vector matrices. ExtractRotationQuat avoids transposing
        # the recorded rotation when converting through scipy for replay.
        self._episode_init_state["object_poses"][obj_path] = {
            "position": self._marker_position(obj_path).tolist(),
            "orientation": self._world_quaternion(obj_path).tolist(),
        }

    def _pipette_axis(self, body: str, tip: str) -> np.ndarray:
        return self.object_utils.get_geometry_center(object_path=body) - self.object_utils.get_geometry_center(
            object_path=tip
        )

    def _grasp_position(self, obj_path: str | None = None) -> np.ndarray:
        obj_path = self.source_obj if obj_path is None else obj_path
        override = getattr(self.cfg.task, "pipette_grasp_local", None)
        if override is None and self.stage.GetPrimAtPath(obj_path + "/Grasp").IsValid():
            return self._marker_position(obj_path + "/Grasp")
        local = getattr(self.cfg.task, "pipette_grasp_local", [0.0, 0.0, 0.265])
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(obj_path))
        return np.asarray(matrix.Transform(Gf.Vec3d(*local)))

    def _slot_position(self) -> np.ndarray:
        selected_slot = getattr(self.cfg.task, "pipette_slot_index", None)
        if selected_slot is not None:
            marker = self.target_obj + f"/Seat{int(selected_slot)}"
            if not self.stage.GetPrimAtPath(marker).IsValid():
                raise ValueError(f"Configured pipette rack slot does not exist: {marker}")
            return self._marker_position(marker)
        if self.stage.GetPrimAtPath(self.target_obj + "/PipetteSeat").IsValid():
            return self._marker_position(self.target_obj + "/PipetteSeat")
        # The rack includes a pre-hung pipette. Its two meshes define the
        # seating height and orientation; the configured local offset selects
        # the adjacent empty fork. Exclude the posts from the placement goal.
        bounds = [
            self.object_utils.get_world_aabb(object_path=self.target_obj + "/group_000/" + name)
            for name in ("mesh_000", "mesh_001")
        ]
        center = (np.minimum(bounds[0]["min"], bounds[1]["min"]) + np.maximum(bounds[0]["max"], bounds[1]["max"])) / 2
        offset = getattr(self.cfg.task, "rack_slot_offset", [-0.30, 0.0, 0.0])
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.target_obj))
        return center + np.asarray(matrix.TransformDir(Gf.Vec3d(*offset)))

    # ------------------------------------------------------------------
    # Geometry-corrected positioning (same as StopperFlaskTask)
    # ------------------------------------------------------------------

    def _compute_geometry_offset(self, obj_path: str) -> np.ndarray:
        geo_center = self.object_utils.get_geometry_center(object_path=obj_path)
        if geo_center is None:
            return np.zeros(3)
        prim = self.stage.GetPrimAtPath(obj_path)
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        if ops:
            raw_translate = np.array(ops[0].Get(), dtype=np.float64)
        else:
            raw_translate = self.object_utils.get_object_xform_position(obj_path)
        return raw_translate - geo_center

    def _randomize_with_geometry_correction(self, obj_path: str, position_range: dict) -> np.ndarray:
        desired = np.array(
            [
                np.random.uniform(position_range["x"][0], position_range["x"][1]),
                np.random.uniform(position_range["y"][0], position_range["y"][1]),
                np.random.uniform(position_range["z"][0], position_range["z"][1]),
            ]
        )
        offset = self._compute_geometry_offset(obj_path)
        self.object_utils.set_object_position(object_path=obj_path, position=desired + offset)
        self._sync_object_rigid_bodies(obj_path)
        self._register_occupied_region(obj_path=obj_path, position=desired)
        self._record_object_pose(obj_path)
        return desired
