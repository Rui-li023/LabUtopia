from pxr import Usd, UsdGeom, Gf, UsdPhysics
from isaacsim.core.utils.stage import get_stage_units
import numpy as np
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
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
            cls._instance = super(ObjectUtils, cls).__new__(cls)
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

    def _get_object_path(self, object_name: str = None, object_path: str = None) -> str:
        if object_path:
            return object_path
        if object_name:
            return f"{self._default_path}/{object_name}"
        raise ValueError("Either object_name or object_path must be provided")

    def get_pick_position(self, object_name: str = None, object_path: str = None) -> np.ndarray:
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

    def get_object_size(self, object_name: str = None, object_path: str = None) -> np.ndarray:
        """Get the world-space size of an object."""
        path = self._get_object_path(object_name, object_path)
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        min_point = bbox.GetRange().GetMin()
        max_point = bbox.GetRange().GetMax()
        world_min_point = np.array([min_point[0], min_point[1], min_point[2], 1.0])
        world_max_point = np.array([max_point[0], max_point[1], max_point[2], 1.0])
        transform_matrix = bbox.GetMatrix()
        transform_matrix_np = np.array(transform_matrix)
        world_size = world_max_point @ transform_matrix_np - world_min_point @ transform_matrix_np
        return world_size[:3]

    def get_object_xform_position(self, object_path: str) -> np.ndarray:
        """Get the world-space position from the object's transform."""
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            print(f"Object at path {object_path} not found.")
            return None

        xformable = UsdGeom.Xformable(prim)
        transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position = transform.ExtractTranslation()
        return np.array(position)
    
    def set_object_position(self, object_path: str, position: np.ndarray, local_position: np.ndarray = None, position_offset: np.ndarray = None) -> None:
        """Set the object's position in world or local space."""
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            print(f"Object at path {object_path} not found.")
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
            
    def get_geometry_center(self, object_name: str = None, object_path: str = None) -> np.ndarray:
        path = self._get_object_path(object_name, object_path)
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        range_3d = bbox.GetRange()
        center = (np.array(range_3d.GetMin()) + np.array(range_3d.GetMax())) / 2.0
        center_hom = np.append(center, 1.0)
        transform = np.array(bbox.GetMatrix())
        world_center = center_hom @ transform
        return world_center[:3]

    
    def get_transform_quat(self, object_path: str, w_first: bool = False) -> np.ndarray:
        """Get the world-space rotation quaternion from the object's transform.
        
        Args:
            object_path: The USD path to the object.
            w_first: If True, return quaternion in [w, x, y, z] format, else [x, y, z, w].
                    Default is False.
        """
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            print(f"Object at path {object_path} not found.")
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
            print(f"Object at path {object_path} not found.")
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
            print(f"Object at path {object_path} not found.")
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
        
    def get_revolute_joint_positions(self, joint_path: str) -> np.ndarray:
        joint_prim = self._stage.GetPrimAtPath(joint_path)

        joint_api = UsdPhysics.Joint(joint_prim)
        body1 = joint_api.GetBody1Rel().GetTargets()
        if not body1:
            print("No body1 found!")
            exit()

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
