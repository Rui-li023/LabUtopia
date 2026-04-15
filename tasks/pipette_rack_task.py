from typing import Any, Dict, Optional

import numpy as np
from pxr import UsdGeom

from .base_task import BaseTask


class PipetteRackTask(BaseTask):
    """Pick pipette from table and place onto pipette rack.

    ``obj_paths[0]`` (pipette) may have complex USD transforms requiring
    geometry correction.  ``obj_paths[1]`` (rack) uses standard positioning.
    """

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()

        self.source_obj = self.cfg.task.obj_paths[0]["path"]  # pipette
        self.target_obj = self.cfg.task.obj_paths[1]["path"]  # rack

        # Pipette needs geometry correction due to pivot/rotation/scale
        self._randomize_with_geometry_correction(
            self.source_obj, self.cfg.task.obj_paths[0]["position_range"]
        )
        # Rack has standard positioning
        self.randomize_object_position(self.target_obj, self.cfg.task.obj_paths[1]["position_range"])

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)
        self.source_obj = self.cfg.task.obj_paths[0]["path"]
        self.target_obj = self.cfg.task.obj_paths[1]["path"]

    def step(self) -> Optional[Dict[str, Any]]:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None
        return self.get_basic_state_info(
            object_path=self.source_obj,
            target_path=self.target_obj,
        )

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
        desired = np.array([
            np.random.uniform(position_range["x"][0], position_range["x"][1]),
            np.random.uniform(position_range["y"][0], position_range["y"][1]),
            np.random.uniform(position_range["z"][0], position_range["z"][1]),
        ])
        offset = self._compute_geometry_offset(obj_path)
        self.object_utils.set_object_position(object_path=obj_path, position=desired + offset)
        self._register_occupied_region(obj_path=obj_path, position=desired)
        self._record_object_pose(obj_path)
        return desired
