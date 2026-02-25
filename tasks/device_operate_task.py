from .base_task import BaseTask


class DeviceOperateTask(BaseTask):
    """Multi-step laboratory device operation task.

    Task sequence:
    1. Open device door
    2. Pick beaker
    3. Place beaker inside device
    4. Close device door
    5. Press device button

    Reads three objects from ``cfg.task.obj_paths``:
    - index 0: source beaker
    - index 1: secondary beaker (beaker3)
    - index 2: target placement object
    """

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.device_path        = cfg.task.device_path
        self.beaker_path        = cfg.task.beaker_path
        self.beaker3_path       = cfg.task.beaker3_path
        self.button_path        = cfg.task.button_path
        self.interior_position  = cfg.task.device_interior_position
        self.beaker3_target_position = cfg.task.beaker3_target_position
        self.initial_beaker_height = None

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        self.source_obj  = self.cfg.task.obj_paths[0]["path"]
        self.beaker3_obj = self.cfg.task.obj_paths[1]["path"]
        self.target_obj  = self.cfg.task.obj_paths[2]["path"]
        self.randomize_object_position(self.source_obj,  self.cfg.task.obj_paths[0]["position_range"])
        self.randomize_object_position(self.beaker3_obj, self.cfg.task.obj_paths[1]["position_range"])
        self.randomize_object_position(self.target_obj,  self.cfg.task.obj_paths[2]["position_range"])

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)
        self.source_obj  = self.cfg.task.obj_paths[0]["path"]
        self.beaker3_obj = self.cfg.task.obj_paths[1]["path"]
        self.target_obj  = self.cfg.task.obj_paths[2]["path"]

    def step(self):
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        button_pressed = self._check_button_pressed()

        return self.get_basic_state_info(
            object_path=self.beaker_path,
            additional_info={
                "door_handle_position":   self.object_utils.get_geometry_center(
                    object_path=f"{self.device_path}/handle"
                ),
                "beaker_position":        self.object_utils.get_geometry_center(object_path=self.beaker_path),
                "beaker_size":            self.object_utils.get_object_size(object_path=self.beaker_path),
                "beaker3_position":       self.object_utils.get_geometry_center(object_path=self.beaker3_path),
                "beaker3_size":           self.object_utils.get_object_size(object_path=self.beaker3_path),
                "beaker3_target_position": self.beaker3_target_position,
                "device_interior_position": self.interior_position,
                "button_position":        self.object_utils.get_geometry_center(object_path=self.button_path),
                "button_pressed":         button_pressed,
                "revolute_joint_position": self.object_utils.get_revolute_joint_positions(
                    joint_path=f"{self.device_path}/RevoluteJoint"
                ),
                "initial_beaker_height":  self.initial_beaker_height,
            },
        )

    def _check_button_pressed(self) -> bool:
        """Return True if the button has been depressed by more than 1 cm."""
        button_z = self.object_utils.get_geometry_center(object_path=self.button_path)[2]
        initial_z = self.cfg.task.button_position[2]
        return button_z < initial_z - 0.01
