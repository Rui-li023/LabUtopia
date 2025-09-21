from .base_task import BaseTask

class SingleObjectTask(BaseTask):
    """
    Base class for handling single target object tasks.
    Suitable for tasks like pick, open, close, etc. that require only one main target object.
    """
    
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        
    def on_task_complete(self, success):
        """
        Handle task completion logic.
        Update object and material indices.
        """
        self.update_object_and_material_indices(success)
        
    def reset(self):
        """
        Reset task state.
        Initialize robot position, apply materials, and place objects.
        """
        super().reset()
        self.robot.initialize()
        
        if self.material_config:
            self.apply_material_to_object(self.material_config.path)
        
        self.current_obj_path = self.place_objects_with_visibility_management(
            self.current_obj_idx, far_distance=10.0
        )
        
    def step(self):
        """
        Execute one simulation step and return current state.
        
        Returns:
            dict: Dictionary containing current state information
        """
        self.frame_idx += 1
        
        if not self.check_frame_limits():
            return None
            
        return self.get_basic_state_info(
            object_path=self.current_obj_path,
            additional_info={
                'object_name': self.current_obj_path.split("/")[-1]
            }
        ) 