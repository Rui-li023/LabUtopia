import numpy as np
from isaacsim.core.utils.prims import get_prim_at_path
from pxr import Gf, UsdGeom, Usd


class Gripper:
    """Kinematically attach an object to the gripper so it follows rigidly.

    Used both at collect time (e.g. the glass rod) and re-created at replay time
    so attached grasps reproduce deterministically.
    """

    def __init__(self):
        self.grasped_object_path = None
        self.gripper_frame_path = None
        self.position_offest = None  # sentinel: None until first update
        self._offset_world = None    # object_world - gripper_world at attach (constant)
        self._bias = None            # translate_op - object_world (constant, handles pivot/parent)

    def reset(self):
        self.release_object()

    def add_object_to_gripper(self, object_path, gripper_frame_path):
        prim = get_prim_at_path(object_path)
        if not prim.IsValid():
            raise ValueError(f"Object at path {object_path} is not valid.")
        self.grasped_object_path = object_path
        self.gripper_frame_path = gripper_frame_path
        self.position_offest = None
        self._offset_world = None
        self._bias = None

    @staticmethod
    def _world_translation(prim):
        return np.array(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation(),
            dtype=np.float64,
        )

    @staticmethod
    def _translate_op(prim):
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and "pivot" not in op.GetOpName():
                return op
        return UsdGeom.Xformable(prim).AddTranslateOp()

    def update_grasped_object_position(self):
        if not self.grasped_object_path or not self.gripper_frame_path:
            return
        gframe = get_prim_at_path(self.gripper_frame_path)
        oprim = get_prim_at_path(self.grasped_object_path)
        if not gframe.IsValid() or not oprim.IsValid():
            return

        gripper_world = self._world_translation(gframe)
        object_world = self._world_translation(oprim)
        translate_op = self._translate_op(oprim)

        if self.position_offest is None:
            # Rigid position offset (object follows the gripper by this constant)
            self._offset_world = object_world - gripper_world
            cur = translate_op.Get()
            cur = np.array([cur[0], cur[1], cur[2]], dtype=np.float64) if cur is not None else object_world
            # Constant bias mapping the desired WORLD origin to the translate op's
            # value (absorbs any pivot/parent offset; for a plain /World prim it's 0).
            self._bias = cur - object_world
            self.position_offest = True

        desired_world = gripper_world + self._offset_world
        new_translate = desired_world + self._bias
        translate_op.Set(Gf.Vec3d(float(new_translate[0]), float(new_translate[1]), float(new_translate[2])))

    def release_object(self):
        self.grasped_object_path = None
        self.gripper_frame_path = None
        self.position_offest = None
        self._offset_world = None
        self._bias = None
