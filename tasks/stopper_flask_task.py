from typing import Any

import numpy as np
from pxr import Gf, UsdGeom

from utils.replay_physics_snapshot import capture_pre_warmup, restore_pre_warmup

from .base_task import BaseTask


class StopperFlaskTask(BaseTask):
    """Pick rubber stopper from table and place into flask opening.

    ``obj_paths[0]`` (stopper) has complex USD transforms requiring geometry
    correction.  ``obj_paths[1]`` (flask) works with standard positioning.
    """

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()

        self.source_obj = self.cfg.task.obj_paths[0]["path"]  # stopper
        self.target_obj = self.cfg.task.obj_paths[1]["path"]  # flask

        # Stopper needs geometry correction due to pivot/rotation/scale
        self._randomize_with_geometry_correction(self.source_obj, self.cfg.task.obj_paths[0]["position_range"])
        # Flask has zero pivot — standard positioning works
        self.randomize_object_position(self.target_obj, self.cfg.task.obj_paths[1]["position_range"])
        if bool(getattr(self.cfg.task, "physical_insertion", False)):
            support = self.cfg.task.obj_paths[2]["path"]
            target = self.object_utils.get_geometry_center(object_path=self.target_obj)
            support_position = np.array([target[0], target[1], self.cfg.task.obj_paths[2]["position_range"]["z"][0]])
            self._place_geometry_center(support, support_position)
            seat = self._marker(support + "/OpenFlaskSeat")
            if seat is None:
                raise ValueError("Stopper insertion requires the original cork support's OpenFlaskSeat marker")
            self._place_geometry_center(self.target_obj, seat + np.array([0, 0, 0.001]))
        if bool(getattr(self.cfg.task, "record_replay_physics_snapshot", False)):
            self._episode_init_state["extra"]["physics_snapshot"] = capture_pre_warmup(self)

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)
        self.source_obj = self.cfg.task.obj_paths[0]["path"]
        self.target_obj = self.cfg.task.obj_paths[1]["path"]
        snapshot = init_state.get("extra", {}).get("physics_snapshot")
        if snapshot is not None:
            restore_pre_warmup(self, snapshot)

    def step(self) -> dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None
        return self.get_basic_state_info(
            object_path=self.source_obj,
            target_path=self.target_obj,
            additional_info={
                "object_tip_position": self._marker(self.source_obj + "/mesh_003/Tip"),
                "object_grasp_position": self._marker(self.source_obj + "/mesh_003/Grasp"),
                "target_mouth_position": self._marker(self.target_obj + "/mesh_001/Mouth"),
                "object_up": self._axis(self.source_obj + "/mesh_003"),
                "target_up": self._axis(self.target_obj + "/mesh_001"),
            },
        )

    def _marker(self, path: str) -> np.ndarray | None:
        prim = self.stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None
        return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation())

    def _axis(self, path: str) -> np.ndarray:
        prim = self.stage.GetPrimAtPath(path)
        axis = UsdGeom.XformCache().GetLocalToWorldTransform(prim).TransformDir(Gf.Vec3d(0, 1, 0))
        vector = np.asarray(axis)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def _place_geometry_center(self, path: str, position: np.ndarray) -> None:
        offset = self._compute_geometry_offset(path)
        self.object_utils.set_object_position(object_path=path, position=position + offset)
        self._sync_object_rigid_bodies(path)
        self._record_object_pose(path)

    # ------------------------------------------------------------------
    # Geometry-corrected positioning (same as FlaskToCorkTask)
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
