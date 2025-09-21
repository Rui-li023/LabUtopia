from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from omni.isaac.sensor import Camera
from utils.object_utils import ObjectUtils
from omni.isaac.core.utils.semantics import add_update_semantics
from utils.camera_utils import process_camera_image
from omni.isaac.core.utils.prims import set_prim_visibility
from pxr import UsdShade

class BaseTask(ABC):
    """
    Base class for all simulation tasks.
    
    Attributes:
        cfg: Task configuration
        world: Simulation world instance
        cameras: Camera settings
        reset_needed (bool): Flag indicating if task needs reset
        frame_idx (int): Current frame index
        object_utils (ObjectUtils): Utility instance for object operations
    """
    def __init__(self, cfg, world, stage, robot):
        """
        Initialize the task.
        
        Args:
            cfg: Task configuration
            world: Simulation world instance
            cameras: Camera settings
        """
        self.cfg = cfg
        self.world = world
        self.stage = stage
        self.robot = robot
        self.reset_needed = False
        self.frame_idx = 0
        self.object_utils = ObjectUtils.get_instance()
        self.setup_cameras()
        self.setup_objects()
        self.setup_materials()
        
        self.current_material_idx = 0
        if len(self.obj_configs) != 0:
            self.episodes_per_obj = int(cfg.max_episodes / len(self.obj_configs))
        else:
            self.episodes_per_obj = 0
        self.current_obj_idx = 0
        self.current_obj_episodes = 0
        
    def reset(self) -> None:
        """
        Reset the task state and simulation world.
        """
        self.world.reset()
        self.reset_needed = False
        self.frame_idx = 0
                
        if self.material_config:
            self.apply_material_to_object(self.material_config.path)
            
    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """
        Execute one step of the task.
        
        Returns:
            Dict[str, Any]: Task step information
        """
        pass

    def get_task_info(self) -> Dict[str, Any]:
        """
        Get task-related information.
        
        Returns:
            Dict[str, Any]: Dictionary containing task information
        """
        return {
            "frame_idx": self.frame_idx,
            "reset_needed": self.reset_needed
        }
        
    def need_reset(self) -> bool:
        """
        Check if task needs to be reset.
        
        Returns:
            bool: True if reset is needed, False otherwise
        """
        return self.reset_needed

    def on_task_complete(self, success: bool) -> None:
        """
        Handle task completion logic.
        Update object and material indices.
        """
        self.update_object_and_material_indices(success)
        self.reset_needed = True
        
    def setup_cameras(self) -> None:
        """
        Set up cameras for the task.
        """
        self.cameras = []
        for cam_cfg in self.cfg.cameras:
            if self.stage.GetPrimAtPath(cam_cfg.prim_path).IsValid():
                camera = Camera(
                    prim_path=cam_cfg.prim_path,
                    name=cam_cfg.name,
                    frequency=60,
                    resolution=tuple(cam_cfg.resolution)
                )
                self.world.scene.add(camera)
            else:
                camera = Camera(
                    prim_path=cam_cfg.prim_path,
                    translation=np.array(cam_cfg.translation),
                    name=cam_cfg.name,
                    frequency=60,
                    resolution=tuple(cam_cfg.resolution)
                )
                camera.set_local_pose(orientation=np.array(cam_cfg.orientation), camera_axes="usd")
                self.world.scene.add(camera)
                camera.set_focal_length(cam_cfg.focal_length)
                
                
            camera.set_clipping_range(near_distance=0.1, far_distance=10.0)
            camera.initialize()
            self.cameras.append(camera)
            
        self.world.reset()
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            camera.initialize()
            
            image_types = cam_cfg.image_type.split('+') if '+' in cam_cfg.image_type else [cam_cfg.image_type]
            
            for image_type in image_types:
                if image_type == "depth":
                    camera.add_distance_to_image_plane_to_frame()
                elif image_type == "pointcloud":
                    camera.add_distance_to_image_plane_to_frame()
                    camera.add_pointcloud_to_frame()
                elif image_type == "segmentation":
                    camera.add_instance_segmentation_to_frame()
                    for class_id, class_to_prim in cam_cfg.class_to_prim.items():
                        e_prim = self.stage.GetPrimAtPath(class_to_prim)
                        add_update_semantics(e_prim, class_id)
                elif image_type == "semantic_pointcloud":
                    camera.add_instance_segmentation_to_frame()
                    camera.add_distance_to_image_plane_to_frame()
                    camera.add_pointcloud_to_frame()
                    for class_id, class_to_prim in cam_cfg.class_to_prim.items():
                        e_prim = self.stage.GetPrimAtPath(class_to_prim)
                        add_update_semantics(e_prim, class_id)
    
    def setup_objects(self) -> None:
        """
        Set up objects in the simulation world.
        """
        self.obj_configs = []
        if hasattr(self.cfg, 'task') and hasattr(self.cfg.task, 'obj_paths'):
            for obj in self.cfg.task.obj_paths:
                if isinstance(obj, str):
                    self.obj_configs.append({
                        'path': obj,
                        'position_range': {
                            'x': [0.24, 0.30],
                            'y': [-0.05, 0.05],
                            'z': [0.85, 0.85]
                        }
                    })
                else:
                    self.obj_configs.append(obj)
                
    def setup_materials(self) -> None:
        """
        Set up materials for the objects.
        """
        self.material_config = None
        self.available_materials = []
        is_infer_mode = hasattr(self.cfg, "mode") and self.cfg.mode == "infer"
        has_infer = hasattr(self.cfg, "infer")
        is_ood = False
        if has_infer and hasattr(self.cfg.infer, "is_test_material"):
            is_ood = bool(self.cfg.infer.is_test_material)
        has_material_paths = hasattr(self.cfg, "task") and hasattr(self.cfg.task, "material_paths") and self.cfg.task.material_paths
        if has_material_paths:
            self.material_config = self.cfg.task.material_paths[0]
            if is_infer_mode and is_ood and hasattr(self.material_config, "test_materials"):
                self.available_materials = getattr(self.material_config, "test_materials", [])
            elif hasattr(self.material_config, "materials"):
                self.available_materials = getattr(self.material_config, "materials", [])
    
    def get_camera_data(self):
        camera_data = {}
        display_data = {}
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
    
    def apply_material_to_object(self, target_path: str, material_idx: int = None) -> None:
        """
        Apply material to the specified object.
        
        Args:
            target_path: Path of the target object
            material_idx: Material index, if None use current material index
        """
        if not self.material_config or not self.available_materials:
            return
            
        if material_idx is None:
            material_idx = self.current_material_idx
            
        target_prim = self.stage.GetPrimAtPath(target_path)
        if target_prim.IsValid():
            material_path = self.available_materials[material_idx]
            mtl_prim = self.stage.GetPrimAtPath(material_path)
            if mtl_prim.IsValid():
                cube_mat_shade = UsdShade.Material(mtl_prim)
                UsdShade.MaterialBindingAPI(target_prim).Bind(
                    cube_mat_shade, 
                    UsdShade.Tokens.strongerThanDescendants
                )
    
    def randomize_object_position(self, obj_path: str, position_range: Dict[str, list]) -> np.ndarray:
        """
        Randomize object position according to position range.
        
        Args:
            obj_path: Object path
            position_range: Position range dictionary containing x, y, z [min, max] values
            
        Returns:
            np.ndarray: Randomly generated position
        """
        position = np.array([
            np.random.uniform(position_range['x'][0], position_range['x'][1]),
            np.random.uniform(position_range['y'][0], position_range['y'][1]),
            np.random.uniform(position_range['z'][0], position_range['z'][1])
        ])
        self.object_utils.set_object_position(object_path=obj_path, position=position)
        return position
    
    def place_objects_with_visibility_management(self, current_obj_idx: int, far_distance: float = 10.0) -> str:
        """
        Place objects and manage visibility, move non-current objects to far distance.
        
        Args:
            current_obj_idx: Index of current object
            far_distance: Far distance where non-current objects are placed
            
        Returns:
            str: Path of current object
        """
        for i, obj_config in enumerate(self.obj_configs):
            obj_path = obj_config['path']
            position_range = obj_config['position_range']
            prim = self.stage.GetPrimAtPath(obj_path)
            
            if prim.IsValid():
                if i == current_obj_idx:
                    self.randomize_object_position(obj_path, position_range)
                    set_prim_visibility(prim, True)
                else:
                    # Move non-current objects to far distance
                    angle = 2 * np.pi * i / len(self.obj_configs)
                    far_position = np.array([
                        far_distance * np.cos(angle),
                        far_distance * np.sin(angle),
                        0.1
                    ])
                    self.object_utils.set_object_position(object_path=obj_path, position=far_position)
                    set_prim_visibility(prim, False)
        
        return self.obj_configs[current_obj_idx]['path']
    
    def get_basic_state_info(self, joint_positions: np.ndarray = None, object_path: str = None, 
                           target_path: str = None, additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get basic state information containing common state data for all tasks.
        
        Args:
            joint_positions: Robot joint positions
            object_path: Object path
            target_path: Target object path
            additional_info: Additional state information
            
        Returns:
            Dict[str, Any]: Basic state information dictionary
        """
        
        if joint_positions is None:
            joint_positions = self.robot.get_joint_positions()
            if joint_positions is None:
                return None
        
        camera_data, display_data = self.get_camera_data()
        state = {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': display_data,
            'done': self.reset_needed,
            'gripper_position': self.robot.get_gripper_position(),
        }
        
        if object_path:
            state.update({
                'object_position': self.object_utils.get_geometry_center(object_path=object_path),
                'object_size': self.object_utils.get_object_size(object_path=object_path),
                'object_path': object_path,
                'object_name': object_path.split("/")[-1]
            })
        
        if target_path:
            state.update({
                'target_position': self.object_utils.get_geometry_center(object_path=target_path),
                'target_size': self.object_utils.get_object_size(object_path=target_path),
                'target_path': target_path,
                'target_name': target_path.split("/")[-1]
            })
        
        if additional_info:
            state.update(additional_info)
            
        return state
    
    def check_frame_limits(self, max_steps: int = None) -> bool:
        """
        Check frame limits, set reset flag if limit is exceeded.
        
        Args:
            max_steps: Maximum steps, if None use value from configuration
            
        Returns:
            bool: Return True if should continue execution, False otherwise
        """
        if self.frame_idx < 5:
            return False
            
        if max_steps is None:
            max_steps = self.cfg.task.max_steps
            
        if self.frame_idx > max_steps:
            self.on_task_complete(True)
            self.reset_needed = True
            
        return True
        
    def update_object_and_material_indices(self, success: bool) -> None:
        """
        Update object and material indices.
        
        Args:
            success: Whether task completed successfully
        """
        if success:
            self.current_obj_episodes += 1
            if self.current_obj_episodes >= self.episodes_per_obj and len(self.obj_configs) > 0:
                self.current_obj_idx = (self.current_obj_idx + 1) % len(self.obj_configs)
                self.current_obj_episodes = 0
            if self.available_materials:
                self.current_material_idx = (self.current_material_idx + 1) % len(self.available_materials)
