from typing import Any

import numpy as np
from pxr import Gf, UsdGeom

from utils.replay_physics_snapshot import capture_pre_warmup, restore_pre_warmup

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

        # Seat height comes from the asset, so the two different supports do
        # not start the flask floating or intersecting the ceramic/cork.
        seat = self._seat_position(self.support_obj)
        if seat is not None:
            self._set_position_with_geometry_correction(self.source_obj, seat + np.array([0, 0, 0.002]))
            self._record_object_pose(self.source_obj)

        # Target: placed separately
        self._randomize_with_geometry_correction(self.target_obj, self.cfg.task.obj_paths[1]["position_range"])
        if bool(getattr(self.cfg.task, "record_replay_physics_snapshot", False)):
            self._episode_init_state["extra"]["physics_snapshot"] = capture_pre_warmup(self)

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)
        self.source_obj = self.cfg.task.obj_paths[0]["path"]
        self.target_obj = self.cfg.task.obj_paths[1]["path"]
        self.support_obj = self.cfg.task.obj_paths[2]["path"]
        snapshot = init_state.get("extra", {}).get("physics_snapshot")
        if snapshot is not None:
            restore_pre_warmup(self, snapshot)
        # NOTE: tried raising gripper↔flask friction (µ=2.0) — collect stayed ~83%
        # and replay dropped to 72%; the round-bottom-flask grasp is marginal for
        # geometric/stability reasons, not friction. Reverted. (ObjectUtils.
        # set_physics_friction kept as a reusable utility.)

    def step(self) -> dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None
        source_prim = self.stage.GetPrimAtPath(self.source_obj)
        # Original lab flask uses local Y as its neck axis; rebuilt assets use Z.
        # Read the asset's frame convention without rotating its visible mesh.
        up_axis = source_prim.GetAttribute("reference:upAxis").Get() or Gf.Vec3f(0, 0, 1)
        return self.get_basic_state_info(
            object_path=self.source_obj,
            target_path=self.target_obj,
            additional_info={
                "target_seat_position": self._seat_position(self.target_obj),
                "object_neck_position": self._marker_position(self.source_obj, "NeckGrasp"),
                "object_up": np.asarray(
                    UsdGeom.XformCache().GetLocalToWorldTransform(source_prim).TransformDir(Gf.Vec3d(*up_axis))
                ),
            },
        )

    def _seat_position(self, obj_path: str) -> np.ndarray | None:
        return self._marker_position(obj_path, "FlaskSeat")

    def _marker_position(self, obj_path: str, marker: str) -> np.ndarray | None:
        prim = self.stage.GetPrimAtPath(obj_path + "/" + marker)
        if not prim.IsValid():
            return None
        return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation())

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
        desired = np.array(
            [
                np.random.uniform(position_range["x"][0], position_range["x"][1]),
                np.random.uniform(position_range["y"][0], position_range["y"][1]),
                np.random.uniform(position_range["z"][0], position_range["z"][1]),
            ]
        )
        self._set_position_with_geometry_correction(obj_path, desired)
        self._register_occupied_region(obj_path=obj_path, position=desired)
        self._record_object_pose(obj_path)
        return desired

    def _set_position_with_geometry_correction(self, obj_path: str, desired_geo_pos: np.ndarray) -> None:
        """Set translate so that geometry centre ends up at *desired_geo_pos*."""
        offset = self._compute_geometry_offset(obj_path)
        corrected = desired_geo_pos + offset
        self.object_utils.set_object_position(object_path=obj_path, position=corrected)
        self._sync_object_rigid_bodies(obj_path)
