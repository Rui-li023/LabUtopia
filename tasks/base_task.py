from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import random
import numpy as np
from isaacsim.sensors.camera import Camera
from utils.object_utils import ObjectUtils
from isaacsim.core.utils.semantics import add_update_semantics
from utils.camera_utils import process_camera_image
from isaacsim.core.utils.prims import set_prim_visibility
from pxr import UsdShade
from loguru import logger

class BaseTask(ABC):
    """
    Base class for all simulation tasks.

    Episode init state is recorded automatically during reset and included in
    every step state dict under the key ``'init_state'``.  This enables
    deterministic replay: pass the recorded ``init_state`` to
    ``reset_with_init_state()`` to restore the exact scene configuration.

    init_state schema::

        {
            "object_poses":     {usd_path: {"position": [x,y,z], "orientation": [x,y,z,w]}},
            "object_materials": {usd_path: material_usd_path},
            "extra":            {task-specific key-value pairs},
        }
    """

    WARMUP_FRAMES: int = 5
    DEFAULT_FAR_DISTANCE: float = 10.0
    DEFAULT_CLIPPING_NEAR: float = 0.1
    DEFAULT_CLIPPING_FAR: float = 10.0

    def __init__(self, cfg: Any, world: Any, stage: Any, robot: Any) -> None:
        """Initialise the task with simulation handles and set up cameras, objects, and materials."""
        self.cfg = cfg
        self.world = world
        self.stage = stage
        self.robot = robot
        self.reset_needed = False
        self.frame_idx = 0
        self.object_utils = ObjectUtils.get_instance()
        self._episode_init_state: Dict = {"object_poses": {}, "object_materials": {}, "extra": {}}

        self.setup_cameras()
        self.setup_objects()
        self.setup_materials()

        self.current_material_idx = 0
        self.episodes_per_obj = int(cfg.max_episodes / len(self.obj_configs)) if self.obj_configs else 0
        self.current_obj_idx = 0
        self.current_obj_episodes = 0

    # -------------------------------------------------------------------------
    # Core lifecycle
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset world state and begin a new episode.

        Clears the episode init state, then re-applies materials (which are
        automatically recorded into the fresh init state).  Subclasses that use
        ``obj_paths`` should call ``self._record_all_config_poses()`` at the
        end of ``reset()`` after placing objects; others may use
        ``self._record_object_pose(path)`` for individual objects.
        """
        self.world.reset()
        self.reset_needed = False
        self.frame_idx = 0
        self._episode_init_state = {"object_poses": {}, "object_materials": {}, "extra": {}}
        self.apply_materials()

    def reset_with_init_state(self, init_state: dict) -> None:
        """Restore scene from a previously recorded initial state.

        Applies saved materials and object poses so the episode starts in
        exactly the same configuration as when data was originally collected.
        Subclasses that manage additional scene state (e.g. object visibility,
        path aliases) should call ``super().reset_with_init_state(init_state)``
        and then restore their own state.

        Args:
            init_state: Dict with keys ``object_poses``, ``object_materials``,
                        ``extra``, ``robot_init_joint_positions``, and
                        ``robot_world_position``.
        """
        self.world.reset()
        self.reset_needed = False
        self.frame_idx = 0
        self._episode_init_state = {
            "object_poses":     dict(init_state.get("object_poses", {})),
            "object_materials": dict(init_state.get("object_materials", {})),
            "extra":            dict(init_state.get("extra", {})),
        }
        for obj_path, material_path in self._episode_init_state["object_materials"].items():
            self._bind_material(obj_path, material_path)
            logger.info(f"Bound material {material_path} to object {obj_path}")
        self._apply_init_state_poses(self._episode_init_state)
        self.robot.initialize()

        # Restore robot joint positions from recorded init state
        robot_joint_positions = init_state.get("robot_init_joint_positions")
        if robot_joint_positions is not None:
            self.robot.set_joint_positions(np.asarray(robot_joint_positions))
            logger.info(f"Restored robot joint positions: {robot_joint_positions}")

        # Restore robot world position from recorded init state
        robot_world_position = init_state.get("robot_world_position")
        if robot_world_position is not None:
            current_orientation = self.robot.get_world_pose()[1]
            self.robot.set_world_pose(position=np.asarray(robot_world_position), orientation=current_orientation)
            logger.info(f"Restored robot world position: {robot_world_position}")

    @abstractmethod
    def step(self) -> Optional[Dict[str, Any]]:
        """Execute one step of the task; returns state dict or None if not ready."""
        pass

    # -------------------------------------------------------------------------
    # Camera setup & data
    # -------------------------------------------------------------------------

    def setup_cameras(self) -> None:
        """Create and initialise all cameras defined in cfg.cameras."""
        self.cameras = []
        for cam_cfg in self.cfg.cameras:
            if self.stage.GetPrimAtPath(cam_cfg.prim_path).IsValid():
                camera = Camera(
                    prim_path=cam_cfg.prim_path,
                    name=cam_cfg.name,
                    frequency=60,
                    resolution=tuple(cam_cfg.resolution),
                )
            else:
                camera = Camera(
                    prim_path=cam_cfg.prim_path,
                    translation=np.array(cam_cfg.translation),
                    name=cam_cfg.name,
                    frequency=60,
                    resolution=tuple(cam_cfg.resolution),
                )
                camera.set_local_pose(orientation=np.array(cam_cfg.orientation), camera_axes="usd")
                camera.set_focal_length(cam_cfg.focal_length)

            clipping = getattr(cam_cfg, "clipping_range", None)
            camera.set_clipping_range(
                near_distance=clipping[0] if clipping else self.DEFAULT_CLIPPING_NEAR,
                far_distance=clipping[1] if clipping else self.DEFAULT_CLIPPING_FAR,
            )
            self.cameras.append(camera)

        self.world.reset()
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            camera.initialize()
            image_types = cam_cfg.image_type.split("+") if "+" in cam_cfg.image_type else [cam_cfg.image_type]
            for image_type in image_types:
                if image_type == "depth":
                    camera.add_distance_to_image_plane_to_frame()
                elif image_type == "pointcloud":
                    camera.add_distance_to_image_plane_to_frame()
                    camera.add_pointcloud_to_frame()
                elif image_type == "segmentation":
                    camera.add_instance_segmentation_to_frame()
                    for class_id, prim_path in cam_cfg.class_to_prim.items():
                        add_update_semantics(self.stage.GetPrimAtPath(prim_path), class_id)
                elif image_type == "semantic_pointcloud":
                    camera.add_instance_segmentation_to_frame()
                    camera.add_distance_to_image_plane_to_frame()
                    camera.add_pointcloud_to_frame()
                    for class_id, prim_path in cam_cfg.class_to_prim.items():
                        add_update_semantics(self.stage.GetPrimAtPath(prim_path), class_id)

    def get_camera_data(self):
        """Return ``(camera_data, display_data)`` dicts for all cameras."""
        camera_data, display_data = {}, {}
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            record, display = process_camera_image(camera, cam_cfg.image_type)
            if record is not None:
                if isinstance(record, dict):
                    for k, v in record.items():
                        camera_data[f"{cam_cfg.name}_{k}"] = v
                else:
                    camera_data[f"{cam_cfg.name}_{cam_cfg.image_type}"] = record
            if display is not None:
                display_data[cam_cfg.name] = display
        return camera_data, display_data

    # -------------------------------------------------------------------------
    # Object & material setup
    # -------------------------------------------------------------------------

    def setup_objects(self) -> None:
        """Populate ``self.obj_configs`` from ``cfg.task.obj_paths``."""
        self.obj_configs = []
        task = getattr(self.cfg, "task", None)
        obj_paths = getattr(task, "obj_paths", None)
        if obj_paths is not None:
            for obj in obj_paths:
                if isinstance(obj, str):
                    self.obj_configs.append({
                        "path": obj,
                        "position_range": {"x": [0.24, 0.30], "y": [-0.05, 0.05], "z": [0.85, 0.85]},
                    })
                else:
                    self.obj_configs.append(obj)

    def setup_materials(self) -> None:
        """Parse material configuration from ``cfg.task.material_paths``.

        Each entry supports:
        - ``path`` / ``paths``: USD path(s) to bind the material to.
        - ``materials``: sequential list used during collection.
        - ``test_materials``: OOD materials used during inference (optional).
        - ``random``: if true, pick randomly each reset instead of cycling.
        """
        self.material_configs: List[Dict] = []
        is_infer = getattr(self.cfg, "mode", None) == "infer"
        infer_cfg = getattr(self.cfg, "infer", None)
        is_ood = is_infer and bool(getattr(infer_cfg, "is_test_material", False))
        task = getattr(self.cfg, "task", None)
        material_paths = getattr(task, "material_paths", None)
        if not material_paths:
            return
        for mat_cfg in material_paths:
            use_test = is_ood and getattr(mat_cfg, "test_materials", None) is not None
            materials = list(mat_cfg.test_materials if use_test else getattr(mat_cfg, "materials", []))
            paths_attr = getattr(mat_cfg, "paths", None)
            path_attr = getattr(mat_cfg, "path", None)
            if paths_attr is not None:
                paths = list(paths_attr)
            elif path_attr is not None:
                paths = [path_attr]
            else:
                paths = []
            self.material_configs.append({
                "paths":    paths,
                "materials": materials,
                "random":   bool(getattr(mat_cfg, "random", False)),
            })

    def apply_materials(self) -> None:
        """Apply configured materials and record them in ``_episode_init_state``."""
        for mat_cfg in self.material_configs:
            if not mat_cfg["materials"]:
                continue
            material_path = (
                random.choice(mat_cfg["materials"])
                if mat_cfg["random"]
                else mat_cfg["materials"][self.current_material_idx % len(mat_cfg["materials"])]
            )
            for obj_path in mat_cfg["paths"]:
                self._bind_material(obj_path, material_path)
                self._episode_init_state["object_materials"][obj_path] = material_path

    def _bind_material(self, obj_path: str, material_path: str) -> None:
        """Bind a USD material to an object prim."""
        target_prim = self.stage.GetPrimAtPath(obj_path)
        if not target_prim.IsValid():
            return
        mtl_prim = self.stage.GetPrimAtPath(material_path)
        if mtl_prim.IsValid():
            UsdShade.MaterialBindingAPI(target_prim).Bind(
                UsdShade.Material(mtl_prim),
                UsdShade.Tokens.strongerThanDescendants,
            )

    # -------------------------------------------------------------------------
    # Object placement helpers
    # -------------------------------------------------------------------------

    def randomize_object_position(self, obj_path: str, position_range: Dict[str, list]) -> np.ndarray:
        """Sample a random position and move the object.

        Poses are recorded in bulk via ``_record_all_config_poses()`` (call at
        end of subclass ``reset()`` after placement).

        Args:
            obj_path: USD prim path of the object.
            position_range: Dict with ``x``, ``y``, ``z`` each as ``[min, max]``.

        Returns:
            The sampled position as a ``(3,)`` array.
        """
        position = np.array([
            np.random.uniform(position_range["x"][0], position_range["x"][1]),
            np.random.uniform(position_range["y"][0], position_range["y"][1]),
            np.random.uniform(position_range["z"][0], position_range["z"][1]),
        ])
        self.object_utils.set_object_position(object_path=obj_path, position=position)
        return position

    def place_objects_with_visibility_management(
        self,
        current_obj_idx: int,
        far_distance: float = None,
        fixed_position: np.ndarray = None,
    ) -> str:
        """Place the active object and hide all others.

        Non-active objects are moved to evenly-spaced far positions and hidden.
        The active object is randomised (or placed at ``fixed_position``) and
        made visible.  Its pose is recorded into ``_episode_init_state``.

        Args:
            current_obj_idx: Index into ``self.obj_configs`` for the active object.
            far_distance: Distance at which inactive objects are placed.
                          Defaults to ``DEFAULT_FAR_DISTANCE``.
            fixed_position: When provided, skip randomisation and place the
                            active object at this exact position (replay mode).

        Returns:
            USD path of the active object.
        """
        if far_distance is None:
            far_distance = self.DEFAULT_FAR_DISTANCE
        for i, obj_cfg in enumerate(self.obj_configs):
            obj_path = obj_cfg["path"]
            prim = self.stage.GetPrimAtPath(obj_path)
            if not prim.IsValid():
                continue
            if i == current_obj_idx:
                if fixed_position is not None:
                    self.object_utils.set_object_position(object_path=obj_path, position=np.array(fixed_position))
                else:
                    self.randomize_object_position(obj_path, obj_cfg["position_range"])
                set_prim_visibility(prim, True)
            else:
                angle = 2 * np.pi * i / len(self.obj_configs)
                far_pos = np.array([far_distance * np.cos(angle), far_distance * np.sin(angle), 0.1])
                self.object_utils.set_object_position(object_path=obj_path, position=far_pos)
                set_prim_visibility(prim, False)
        return self.obj_configs[current_obj_idx]["path"]

    # -------------------------------------------------------------------------
    # Init state recording / restoration helpers
    # -------------------------------------------------------------------------

    def _record_object_pose(self, obj_path: str) -> None:
        """Snapshot the current world pose of ``obj_path`` into init state."""
        pose = self.object_utils.get_world_pose(obj_path)
        if pose is not None:
            self._episode_init_state["object_poses"][obj_path] = {
                "position":    pose["position"].tolist(),
                "orientation": pose["orientation"].tolist(),
            }

    def _record_all_config_poses(self) -> None:
        """Record current world pose for every object in cfg.task.obj_paths.

        Call this at the end of subclass ``reset()`` after placing objects.
        Materials are already recorded in ``apply_materials()`` from
        cfg.task.material_paths.
        """
        for obj_cfg in self.obj_configs:
            path = obj_cfg["path"]
            pose = self.object_utils.get_world_pose(path)
            if pose is not None:
                self._episode_init_state["object_poses"][path] = {
                    "position":    pose["position"].tolist(),
                    "orientation": pose["orientation"].tolist(),
                }

    def _apply_init_state_poses(self, init_state: dict, restore_orientation: bool = False) -> None:
        """Restore object world poses from ``init_state['object_poses']``.

        Args:
            init_state: Dict containing ``object_poses`` with position and orientation.
            restore_orientation: If True, restore orientation; if False, only restore position.
        """
        for path, pose in init_state.get("object_poses", {}).items():
            position = np.asarray(pose["position"])
            if restore_orientation and "orientation" in pose:
                self.object_utils.set_world_pose(
                    path,
                    position,
                    np.asarray(pose["orientation"]),
                )
                logger.info(f"Restored object {path} to position {pose['position']} and orientation {pose['orientation']}")
            else:
                self.object_utils.set_object_position(object_path=path, position=position)
                logger.info(f"Restored object {path} to position {pose['position']}")

    # -------------------------------------------------------------------------
    # Step state helpers
    # -------------------------------------------------------------------------

    def get_basic_state_info(
        self,
        joint_positions: np.ndarray = None,
        object_path: str = None,
        target_path: str = None,
        additional_info: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build the common step state dict shared across all tasks.

        Always includes ``init_state`` so controllers / collectors can record
        the episode's initial configuration without extra bookkeeping.

        Args:
            joint_positions: Override robot joint positions (fetched if None).
            object_path: Primary object; adds ``object_position``, ``object_size``,
                         ``object_path``, ``object_name``.
            target_path: Target object; adds ``target_position``, ``target_size``,
                         ``target_path``, ``target_name``.
            additional_info: Extra key-value pairs merged into the state dict.

        Returns:
            State dict, or ``None`` if joint positions are unavailable.
        """
        if joint_positions is None:
            joint_positions = self.robot.get_joint_positions()
            if joint_positions is None:
                return None

        camera_data, display_data = self.get_camera_data()
        state = {
            "joint_positions":  joint_positions,
            "camera_data":      camera_data,
            "camera_display":   display_data,
            "done":             self.reset_needed,
            "gripper_position": self.robot.get_gripper_position(),
            "init_state":       self._episode_init_state,
        }

        if object_path:
            state.update({
                "object_position": self.object_utils.get_geometry_center(object_path=object_path),
                "object_size":     self.object_utils.get_object_size(object_path=object_path),
                "object_path":     object_path,
                "object_name":     object_path.split("/")[-1],
            })

        if target_path:
            state.update({
                "target_position": self.object_utils.get_geometry_center(object_path=target_path),
                "target_size":     self.object_utils.get_object_size(object_path=target_path),
                "target_path":     target_path,
                "target_name":     target_path.split("/")[-1],
            })

        if additional_info:
            state.update(additional_info)

        return state

    def check_frame_limits(self, max_steps: int = None) -> bool:
        """Guard for warm-up frames and episode length.

        Returns ``False`` for the first 5 frames (warm-up) and triggers
        ``on_task_complete`` once ``frame_idx`` exceeds ``max_steps``.

        Args:
            max_steps: Episode length cap; defaults to ``cfg.task.max_steps``.

        Returns:
            ``False`` during warm-up, ``True`` otherwise.
        """
        if self.frame_idx < self.WARMUP_FRAMES:
            return False
        if max_steps is None:
            max_steps = getattr(getattr(self.cfg, "task", None), "max_steps", float("inf"))
        logger.info(f"Frame idx: {self.frame_idx}, max steps: {max_steps}")
        
        if self.frame_idx > max_steps:
            self.on_task_complete(True)
        return True

    # -------------------------------------------------------------------------
    # Misc public API
    # -------------------------------------------------------------------------

    def get_task_info(self) -> Dict[str, Any]:
        return {"frame_idx": self.frame_idx, "reset_needed": self.reset_needed}

    def need_reset(self) -> bool:
        return self.reset_needed

    # -------------------------------------------------------------------------
    # Task completion & index management
    # -------------------------------------------------------------------------

    def on_task_complete(self, success: bool) -> None:
        """Update object/material indices and set the reset flag."""
        self.update_object_and_material_indices(success)
        self.reset_needed = True

    def update_object_and_material_indices(self, success: bool) -> None:
        """Advance object and material cycling indices on successful episodes."""
        if not success:
            return
        self.current_obj_episodes += 1
        if self.current_obj_episodes >= self.episodes_per_obj and self.obj_configs:
            self.current_obj_idx = (self.current_obj_idx + 1) % len(self.obj_configs)
            self.current_obj_episodes = 0
        sequential = [mc for mc in self.material_configs if not mc["random"] and mc["materials"]]
        if sequential:
            self.current_material_idx = (self.current_material_idx + 1) % len(sequential[0]["materials"])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(frame={self.frame_idx}, "
            f"objs={len(self.obj_configs)}, mats={len(self.material_configs)})"
        )
