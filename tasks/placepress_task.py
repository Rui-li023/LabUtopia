import numpy as np
from tasks.base_task import BaseTask

class PlacePressTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        """Initialize the Place and Press task.

        Args:
            cfg: Configuration object for the task.
            world: The simulation world instance.
            stage: The USD stage for the simulation.
            robot: The robot instance used in the task.
        """
        super().__init__(cfg, world, stage, robot)

        self.source_beaker = self.cfg.task.obj_paths[0]['path']
        self.target_plat = self.cfg.task.obj_paths[1]['path']
        self.target_sub_plat = self.cfg.task.obj_paths[1]['sub_path']
        self.button = self.cfg.task.obj_paths[2]['path']

    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()

        obj_position_range = self.cfg.task.obj_paths[0]['position_range'] 
        object_position = np.array([
                        np.random.uniform(obj_position_range['x'][0], obj_position_range['x'][1]),
                        np.random.uniform(obj_position_range['y'][0], obj_position_range['y'][1]),
                        obj_position_range['z'][0]
                    ])
        self.object_utils.set_object_position(object_path=self.source_beaker, position=object_position)

        target_position_range = self.cfg.task.obj_paths[1]['position_range']
        target_position = np.array([
                        np.random.uniform(target_position_range['x'][0], target_position_range['x'][1]),
                        np.random.uniform(target_position_range['y'][0], target_position_range['y'][1]),
                        target_position_range['z'][0]
                    ])
        self.object_utils.set_object_position(object_path=self.target_plat, position=target_position)

    def step(self):
        """Execute one simulation step.

        Returns:
            dict: A dictionary containing simulation state data, or None if not ready.
        """
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None
        
        target_position = self.object_utils.get_object_xform_position(object_path=self.target_sub_plat)
        button_position = self.object_utils.get_object_xform_position(object_path=self.button)

        target_position[2] += 0.045

        return self.get_basic_state_info(
            object_path=self.source_beaker,
            additional_info={
                'source_beaker': self.source_beaker,
                'target_position': target_position,
                'target_name': self.target_plat.split("/")[-1],
                'target_path': self.target_plat,
                'button_position': button_position,
            }
        )
