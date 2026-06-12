from typing import Any, Dict, Optional

import numpy as np
from pxr import UsdGeom

from .base_task import BaseTask


class FlaskToCorkTask(BaseTask):
    """Level-2 pick-and-place with three objects.

    - ``obj_paths[0]``: source object (flask) — randomised, picked up.
    - ``obj_paths[1]``: target object — randomised separately, flask placed here.
    - ``obj_paths[2]``: support object — placed under the flask at start (same XY).

    The cork ring and pipeclay triangle have complex USD transforms (pivot +
    rotation + scale) that cause their geometry centres to be far from their
    translate values.  This task computes a correction offset so that
    positioning places the geometry where intended.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()

        self.source_obj = self.cfg.task.obj_paths[0]["path"]
        self.target_obj = self.cfg.task.obj_paths[1]["path"]
        self.support_obj = self.cfg.task.obj_paths[2]["path"]

        # Source (flask): randomize on table
        flask_pos = self.randomize_object_position(self.source_obj, self.cfg.task.obj_paths[0]["position_range"])

        # Support: placed under the flask at start (same XY, its own Z)
        support_z = self.cfg.task.obj_paths[2]["position_range"]["z"][0]
        support_pos = np.array([flask_pos[0], flask_pos[1], support_z])
        self._set_position_with_geometry_correction(self.support_obj, support_pos)
        self._record_object_pose(self.support_obj)

        # Target: placed separately
        self._randomize_with_geometry_correction(
            self.target_obj, self.cfg.task.obj_paths[1]["position_range"]
        )

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)
        self.source_obj = self.cfg.task.obj_paths[0]["path"]
        self.target_obj = self.cfg.task.obj_paths[1]["path"]
        self.support_obj = self.cfg.task.obj_paths[2]["path"]
        # NOTE: tried raising gripper↔flask friction (µ=2.0) — collect stayed ~83%
        # and replay dropped to 72%; the round-bottom-flask grasp is marginal for
        # geometric/stability reasons, not friction. Reverted. (ObjectUtils.
        # set_physics_friction kept as a reusable utility.)

    def step(self) -> Optional[Dict[str, Any]]:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None
        return self.get_basic_state_info(
            object_path=self.source_obj,
            target_path=self.target_obj,
        )

    # ------------------------------------------------------------------
    # Geometry-corrected positioning helpers
    # ------------------------------------------------------------------

    def _compute_geometry_offset(self, obj_path: str) -> np.ndarray:
        """Return the vector from geometry centre to the raw translate op value.

        Adding this offset to a desired geometry position yields the translate
        value that will place the geometry there.
        """
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
        """Sample a random position and place the object so its geometry centre is there."""
        desired = np.array([
            np.random.uniform(position_range["x"][0], position_range["x"][1]),
            np.random.uniform(position_range["y"][0], position_range["y"][1]),
            np.random.uniform(position_range["z"][0], position_range["z"][1]),
        ])
        self._set_position_with_geometry_correction(obj_path, desired)
        self._register_occupied_region(obj_path=obj_path, position=desired)
        self._record_object_pose(obj_path)
        return desired

    def _set_position_with_geometry_correction(self, obj_path: str, desired_geo_pos: np.ndarray) -> None:
        """Set translate so that geometry centre ends up at *desired_geo_pos*."""
        offset = self._compute_geometry_offset(obj_path)
        corrected = desired_geo_pos + offset
        self.object_utils.set_object_position(object_path=obj_path, position=corrected)
