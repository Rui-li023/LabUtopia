import numpy as np
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.stage import get_stage_units
from loguru import logger
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation as R


class ObjectUtils:
    _instance = None

    @classmethod
    def get_instance(cls, stage: Usd.Stage = None, default_path: str = "/World") -> "ObjectUtils":
        if cls._instance is None:
            if stage is None:
                raise ValueError("Stage must be provided for first instance")
            cls._instance = cls(stage, default_path)
        return cls._instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, stage: Usd.Stage, default_path: str = "/World"):
        if not hasattr(self, '_initialized'):
            self._stage = stage
            self._default_path = default_path
            self._pick_height_offsets = {
                "rod": 0.04,
                "tube": 0.01,
                "beaker": 0.0,
                "erlenmeyer flask": 0.018,
                "cylinder": 0.0,
                "petri dish": 0.005,
                "pipette": 0.008,
                "microscope slide": 0.002
            }
            self._initialized = True

    def _get_object_path(self, object_name: str | None = None, object_path: str | None = None) -> str:
        if object_path:
            return object_path
        if object_name:
            return f"{self._default_path}/{object_name}"
        raise ValueError("Either object_name or object_path must be provided")

    def get_pick_position(self, object_name: str | None = None, object_path: str | None = None) -> np.ndarray:
        """Get the object's pick position with height offset."""
        position = self.get_geometry_center(object_name, object_path)
        if position is None:
            return None

        name = object_name or object_path.split('/')[-1]
        for key, offset in self._pick_height_offsets.items():
            if key in name.lower():
                position[2] += offset / get_stage_units()
                return position

        position[2] += 0.02 / get_stage_units()
        return position

    def get_object_size(self, object_name: str | None = None, object_path: str | None = None) -> np.ndarray:
        """Get the world-space size of an object."""
        aabb = self.get_world_aabb(object_name=object_name, object_path=object_path)
        if aabb is None:
            return None
        return aabb["size"]

    def get_world_aabb(self, object_name: str | None = None, object_path: str | None = None) -> dict | None:
        """Get the object's world-aligned bounding box."""
        path = self._get_object_path(object_name, object_path)
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        aligned_range = bbox.ComputeAlignedRange() if hasattr(bbox, "ComputeAlignedRange") else bbox.GetRange()
        min_point = np.array(aligned_range.GetMin(), dtype=np.float64)
        max_point = np.array(aligned_range.GetMax(), dtype=np.float64)
        size = max_point - min_point
        center = (min_point + max_point) / 2.0
        return {
            "min": min_point,
            "max": max_point,
            "size": size,
            "center": center,
        }

    def get_support_offset_z(self, object_name: str | None = None, object_path: str | None = None) -> float | None:
        """Return the z offset from the prim xform to the object's supporting bottom."""
        aabb = self.get_world_aabb(object_name=object_name, object_path=object_path)
        if aabb is None:
            return None

        path = self._get_object_path(object_name, object_path)
        xform_position = self.get_object_xform_position(path)
        if xform_position is None:
            return None
        return float(xform_position[2] - aabb["min"][2])

    def get_surface_top_z(self, object_name: str | None = None, object_path: str | None = None) -> float | None:
        """Return the top z value of a supporting surface prim."""
        aabb = self.get_world_aabb(object_name=object_name, object_path=object_path)
        if aabb is None:
            return None
        return float(aabb["max"][2])

    def get_object_xform_position(self, object_path: str) -> np.ndarray:
        """Get the world-space position from the object's transform."""
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            logger.warning(f"Object at path {object_path} not found.")
            return None

        xformable = UsdGeom.Xformable(prim)
        transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position = transform.ExtractTranslation()
        return np.array(position)

    def set_object_scale(self, object_path: str, scale) -> np.ndarray | None:
        """Set the object's local scale. Scalar or per-axis ``(3,)``.

        Writes ``xformOp:scale`` on the prim, reusing the existing op when there
        is one — appending a second scale op would silently multiply with the
        authored one and drift a little further every episode.

        Note the collider does NOT re-cook: Isaac keeps the approximation built
        at load time, so a scaled mesh grasps against slightly stale collision
        geometry. Fine for the modest (~0.9x) squashes used to match a real
        object's height; do not rely on it for large changes.
        """
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            logger.warning(f"Object at path {object_path} not found; scale ignored.")
            return None

        s = np.asarray(scale, dtype=float).reshape(-1)
        if s.size == 1:
            s = np.repeat(s, 3)
        if s.size != 3:
            logger.warning(f"scale for {object_path} must be scalar or 3 values, got {s.tolist()}")
            return None

        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                op.Set(Gf.Vec3d(*s))
                return s
        xformable.AddScaleOp().Set(Gf.Vec3d(*s))
        return s

    def set_object_position(self, object_path: str, position: np.ndarray, local_position: np.ndarray = None, position_offset: np.ndarray = None) -> None:
        """Set the object's position in world or local space."""
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            logger.warning(f"Object at path {object_path} not found.")
            return

        xformable = UsdGeom.Xformable(prim)
        xform_ops = xformable.GetOrderedXformOps()
        if local_position is not None and position_offset is not None:
            new_position = Gf.Vec3d(*(local_position + position_offset).astype(np.float64))
        else:
            new_position = Gf.Vec3d(*np.asarray(position, dtype=np.float64))

        if xform_ops:
            xform_ops[0].Set(new_position)
        else:
            xformable.AddTranslateOp().Set(new_position)

    def set_physics_friction(self, object_path: str, static_friction: float = 1.5,
                             dynamic_friction: float = 1.5, restitution: float = 0.0) -> None:
        """Bind a high-friction PhysX material to an object's collision geometry.

        A marginal grasp slips because the gripper-object contact friction is too
        low; raising it to a realistic value makes the grasp robust in both
        collect and replay (fixes a scene-physics misalignment, not gaming).
        Idempotent: one shared material per friction value, bound to the prim and
        its descendants for the ``physics`` purpose.
        """
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim or not prim.IsValid():
            logger.warning(f"set_physics_friction: prim {object_path} invalid")
            return
        tag = f"{static_friction}_{dynamic_friction}".replace(".", "p").replace("-", "n")
        mat_path = f"/World/PhysicsMaterials/friction_{tag}"
        mat_prim = self._stage.GetPrimAtPath(mat_path)
        if not mat_prim or not mat_prim.IsValid():
            if not self._stage.GetPrimAtPath("/World/PhysicsMaterials").IsValid():
                UsdGeom.Scope.Define(self._stage, "/World/PhysicsMaterials")
            material = UsdShade.Material.Define(self._stage, mat_path)
            UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api = UsdPhysics.MaterialAPI(material.GetPrim())
            api.CreateStaticFrictionAttr().Set(float(static_friction))
            api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
            api.CreateRestitutionAttr().Set(float(restitution))
            mat_prim = material.GetPrim()
        material = UsdShade.Material(mat_prim)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
        logger.info(f"set_physics_friction: bound µ={static_friction} to {object_path}")

    def set_object_mass(self, object_path: str, mass_kg: float | None = None,
                        scale: float | None = None) -> float | None:
        """Override a rigid body's mass (kg) at runtime.

        Several graspable glassware meshes have an inflated ``physics:density``
        (e.g. beaker2 = 12 vs 4-5 for the cylinders) that was raised to keep the
        binary slam-close from ejecting them; the heavy mass then introduces
        inertial wobble during the pour tilt. This lets a task lower the source
        mass for a clean collect once the grasp uses a distance/force hold.

        Either ``mass_kg`` (absolute) or ``scale`` (multiply the body's current
        PhysX mass) may be given. Authors ``physics:mass`` so a re-parse keeps the
        value, and pushes it to the live body via SingleRigidPrim when physics is
        already initialized. Returns the resulting mass, or None on failure.
        """
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim or not prim.IsValid():
            logger.warning(f"set_object_mass: prim {object_path} invalid")
            return None
        rp = None
        try:
            from isaacsim.core.prims import SingleRigidPrim
            rp = SingleRigidPrim(object_path)
        except Exception as exc:  # physics view not ready yet — fall back to USD attr
            logger.info(f"set_object_mass: runtime view unavailable ({exc}); authoring physics:mass")
        target = mass_kg
        if target is None and scale is not None:
            # Scale from the ORIGINAL authored mass, cached on first call — scaling
            # the live (already-overridden) mass every reset compounds toward zero.
            if not hasattr(self, "_original_masses"):
                self._original_masses = {}
            base = self._original_masses.get(object_path)
            if base is None:
                base = None
                if rp is not None:
                    try:
                        base = float(rp.get_mass())
                    except Exception:
                        base = None
                if base is None or base <= 0.0:
                    logger.warning(f"set_object_mass: cannot read current mass of {object_path}; "
                                   f"scale={scale} ignored")
                    return None
                self._original_masses[object_path] = base
            target = base * float(scale)
        if target is None:
            return None
        # Author physics:mass (precedence over density) and neutralise density.
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetMassAttr().Set(float(target))
        dens = prim.GetAttribute("physics:density")
        if dens and dens.IsValid() and dens.HasAuthoredValue():
            dens.Set(0.0)
        # Push to the live body so the change takes effect this episode.
        if rp is not None:
            try:
                rp.set_mass(float(target))
                target = float(rp.get_mass())
            except Exception as exc:
                logger.info(f"set_object_mass: live set deferred ({exc})")
        logger.info(f"set_object_mass: {object_path} -> {target:.4f} kg")
        return target

    def get_geometry_center(self, object_name: str | None = None, object_path: str | None = None) -> np.ndarray:
        aabb = self.get_world_aabb(object_name=object_name, object_path=object_path)
        if aabb is None:
            return None
        return aabb["center"].copy()

    def get_transform_quat(self, object_path: str, w_first: bool = False) -> np.ndarray:
        """Get the world-space rotation quaternion from the object's transform.

        Args:
            object_path: The USD path to the object.
            w_first: If True, return quaternion in [w, x, y, z] format, else [x, y, z, w].
                    Default is False.
        """
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            logger.warning(f"Object at path {object_path} not found.")
            return None

        rotation = prim.GetAttribute("xformOp:orient").Get()

        if rotation is None:
            rotation = prim.GetAttribute("xformOp:rotateXYZ").Get()
            rotation = euler_angles_to_quats(rotation, degrees=True)
            # rotation is already in [w, x, y, z] format
            return np.array([rotation[0], rotation[1], rotation[2], rotation[3]]) if w_first else np.array([rotation[1], rotation[2], rotation[3], rotation[0]])

        quat = np.array([rotation.GetImaginary()[0], rotation.GetImaginary()[1], rotation.GetImaginary()[2], rotation.GetReal()])
        if abs(quat[0]) > 0.5 and abs(quat[0]) > abs(quat[3]):
            quat = np.array([quat[1], quat[2], quat[3], quat[0]])

        return np.array([quat[3], quat[0], quat[1], quat[2]]) if w_first else quat

    def get_world_pose(self, object_path: str) -> dict:
        """Get world-space position and orientation (quaternion) for the object.

        Returns:
            dict: {"position": np.ndarray (3,), "orientation": np.ndarray (4,) [x,y,z,w]}
                  or None if prim invalid.
        """
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            logger.warning(f"Object at path {object_path} not found.")
            return None
        xformable = UsdGeom.Xformable(prim)
        transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position = np.array(transform.ExtractTranslation())
        rot_matrix = np.array([
            [transform[0][0], transform[0][1], transform[0][2]],
            [transform[1][0], transform[1][1], transform[1][2]],
            [transform[2][0], transform[2][1], transform[2][2]],
        ])
        quat = R.from_matrix(rot_matrix).as_quat()
        return {"position": position, "orientation": quat}

    def set_world_pose(self, object_path: str, position: np.ndarray, orientation: np.ndarray) -> None:
        """Set the object's local position and orientation (used for restoration when parent is identity).

        Args:
            object_path: USD path to the prim.
            position: (3,) world position.
            orientation: (4,) quaternion [x, y, z, w] in scipy format.
        """
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            logger.warning(f"Object at path {object_path} not found.")
            return
        xformable = UsdGeom.Xformable(prim)
        xform_ops = xformable.GetOrderedXformOps()
        pos_vec = Gf.Vec3d(*np.asarray(position, dtype=np.float64))
        quat = np.asarray(orientation, dtype=np.float64)
        if len(quat) == 4 and (quat[3] >= -1.1 and quat[3] <= 1.1):
            rot = Gf.Quatf(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))
        else:
            rot = Gf.Quatf(1, 0, 0, 0)
        for op in xform_ops:
            op_name = op.GetOpType()
            if op_name == UsdGeom.XformOp.TypeTranslate:
                op.Set(pos_vec)
                break
        else:
            xformable.AddTranslateOp().Set(pos_vec)
        for op in xform_ops:
            op_name = op.GetOpType()
            if op_name == UsdGeom.XformOp.TypeOrient:
                op.Set(rot)
                return
            if op_name == UsdGeom.XformOp.TypeRotateXYZ:
                euler = R.from_quat(quat).as_euler("xyz", degrees=True)
                op.Set(Gf.Vec3f(*euler))
                return
        xformable.AddOrientOp().Set(rot)

    def _joint_attr(self, joint_path: str, attr_name: str):
        joint_prim = self._stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            logger.error(f"Joint prim not found: {joint_path}")
            return None
        attr = joint_prim.GetAttribute(attr_name)
        if not attr or not attr.IsValid():
            logger.error(f"{attr_name} attr missing on {joint_path}")
            return None
        return attr

    def set_joint_local_pos(self, joint_path: str, local_pos: np.ndarray, side: int = 0) -> None:
        """Set ``physics:localPos{side}`` on a USD physics joint."""
        attr = self._joint_attr(joint_path, f"physics:localPos{side}")
        if attr is None:
            return
        lp = np.asarray(local_pos, dtype=np.float32)
        attr.Set(Gf.Vec3f(float(lp[0]), float(lp[1]), float(lp[2])))

    def get_joint_local_pos(self, joint_path: str, side: int = 0) -> np.ndarray:
        attr = self._joint_attr(joint_path, f"physics:localPos{side}")
        if attr is None:
            return None
        v = attr.Get()
        return np.array([v[0], v[1], v[2]], dtype=np.float32) if v is not None else None

    def get_joint_bodies(self, joint_path: str):
        joint_prim = self._stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            return None, None
        j = UsdPhysics.Joint(joint_prim)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        b0p = str(b0[0]) if b0 else None
        b1p = str(b1[0]) if b1 else None
        return b0p, b1p

    # Backward-compat shims
    def set_joint_local_pos0(self, joint_path: str, local_pos: np.ndarray) -> None:
        self.set_joint_local_pos(joint_path, local_pos, side=0)

    def get_joint_local_pos0(self, joint_path: str) -> np.ndarray:
        return self.get_joint_local_pos(joint_path, side=0)

    def get_revolute_joint_positions(self, joint_path: str) -> np.ndarray:
        joint_prim = self._stage.GetPrimAtPath(joint_path)

        joint_api = UsdPhysics.Joint(joint_prim)
        body1 = joint_api.GetBody1Rel().GetTargets()
        if not body1:
            logger.error("No body1 found for joint!")
            return None

        body1_prim = self._stage.GetPrimAtPath(body1[0])

        body1_xform = UsdGeom.Xformable(body1_prim)
        body1_world_transform = Gf.Matrix4f(body1_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

        local_pos1 = joint_api.GetLocalPos1Attr().Get()
        local_rot1 = joint_api.GetLocalRot1Attr().Get()

        rotation_matrix = Gf.Matrix3f(local_rot1)
        local_transform = Gf.Matrix4f()
        local_transform.SetTranslateOnly(local_pos1)
        local_transform.SetRotateOnly(rotation_matrix)
        joint_position = local_transform * body1_world_transform
        return np.array(joint_position.ExtractTranslation())
