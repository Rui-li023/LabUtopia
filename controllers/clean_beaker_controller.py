import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.pour_controller import PourController
from .atomic_actions.shake_controller import ShakeController

class CleanBeakerTaskController(BaseController):
    """
    Controller for clean beaker tasks with two operation modes:
    - Collection mode: Gathers training data through demonstrations
    - Inference mode: Executes learned policies for autonomous cleaning

    Attributes:
        mode (str): Operation mode ("collect" or "infer")
        _current_step (int): Current step in the task sequence
        frame_count (int): Frame counter for episode management
    """
    
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self._current_step = 1
        self.frame_count = 0
        
        if self.mode == "collect":
            self._init_collect_mode(cfg, robot)
        else:
            self._init_infer_mode(cfg, robot)
    
    def _init_collect_mode(self, cfg, robot):
        """
        Initializes components for data collection mode.
        Sets up atomic action controllers and data collector.

        Args:
            cfg: Configuration object containing collection settings
            robot: Robot instance to control
        """
        super()._init_collect_mode(cfg, robot)

        # 1. Pick beaker2
        self.pick_beaker2 = PickController(
            name="pick_beaker2",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 1, 0.05, 0.004, 1]
        )

        # 2. Pour beaker2 to beaker1
        self.pour_beaker2 = PourController(
            name="pour_beaker2",
            cspace_controller=self.rmp_controller,
            events_dt=[0.006, 0.005, 0.009, 0.05, 0.009, 1]
        )

        # 3. Place beaker2 to plat2
        self.place_beaker2 = PlaceController(
            name="place_beaker2",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.008, 1, 0.05, 0.01, 1],
            robot=robot,
        )

        # 4. Pick beaker1
        self.pick_beaker1 = PickController(
            name="pick_beaker1",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 1, 0.05, 0.004, 1]
        )

        # 5. Shake beaker1
        self.shake_beaker1 = ShakeController(
            name="shake_beaker1",
            cspace_controller=self.rmp_controller
        )

        # 6. Pour beaker1 to target_beaker
        self.pour_beaker1 = PourController(
            name="pour_beaker1",
            cspace_controller=self.rmp_controller,
            events_dt=[0.006, 0.005, 0.009, 0.05, 0.009, 1]
        )

        # 7. Place beaker1 to plat1
        self.place_beaker1 = PlaceController(
            name="place_beaker1",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.008, 1, 0.05, 0.01, 1],
            robot=robot,
        )

    def reset(self):
        super().reset()
        
        if self.mode == "collect":
            self.pick_beaker2.reset()
            self.pour_beaker2.reset()
            self.place_beaker2.reset()
            self.pick_beaker1.reset()
            self.shake_beaker1.reset()
            self.pour_beaker1.reset()
            self.place_beaker1.reset()
        elif self.mode == "infer":
            self.inference_engine.reset()
        
        self._current_step = 1
        self.frame_count = 0
        self._logged_step_1 = False

    def step(self, state):
        if self.mode == "collect":
            return self._step_collect(state)
        elif self.mode == "replay":
            return self._step_replay(state)
        else:
            return self._step_infer(state)
    
    def _step_collect(self, state):
        """
        Executes one step in collection mode.
        Records demonstrations and manages episode transitions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        action = None
        record_array = None
        done = False
        success = False

        if self._current_step == 1:
            if not getattr(self, "_logged_step_1", False):
                print(f"[cleanbeaker] step 1 (pick beaker2) begin; beaker_2_pos={state['beaker_2_position'].tolist() if hasattr(state['beaker_2_position'], 'tolist') else state['beaker_2_position']}")
                self._logged_step_1 = True
            # 1. Pick beaker2
            action, record_array = self.pick_beaker2.forward(
                picking_position=state['beaker_2_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="beaker_l",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                pre_offset_x=0.05,
                pre_offset_z=0.05,
                gripper_distances=0.027
            )
            if self.pick_beaker2.is_done():
                print(f"[cleanbeaker] step 1 (pick beaker2) done")
                self._current_step = 2

        elif self._current_step == 2:
            # 2. Pour beaker2 to beaker1
            action, record_array = self.pour_beaker2.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=state['object_size'],
                target_position=state['beaker_1_position'],
                gripper_position=state['gripper_position'],
                source_name="beaker",
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
            )
            if self.pour_beaker2.is_done():
                print(f"[cleanbeaker] step 2 (pour beaker2→beaker1) done; beaker2_pos={state['beaker_2_position'].tolist() if hasattr(state['beaker_2_position'], 'tolist') else state['beaker_2_position']}")
                self._current_step = 3

        elif self._current_step == 3:
            # 3. Place beaker2 to plat2
            # Aim the plat centre with the stock 0.027 grip: the released
            # beaker always settles FLAT at dx=0.0403±0.0003 (dz=0.0358, fully
            # on the platform). Every attempt to cancel that settle shift —
            # offset aim points (-0.01/-0.02/-0.04) or a firmer grip (0.024) —
            # moved the drop dynamics off the flat sweet spot and the beaker
            # tipped on the plat edge 40-50% of the time (dz 0.049-0.056).
            # The deterministic flat landing is the right behaviour; the
            # success box in _check_success accounts for the settle shift.
            action, record_array = self.place_beaker2.forward(
                place_position=state['plat_2_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 40])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_beaker2.is_done():
                print(f"[cleanbeaker] step 3 (place beaker2→plat2) done; beaker2_pos={state['beaker_2_position'].tolist() if hasattr(state['beaker_2_position'], 'tolist') else state['beaker_2_position']} plat2={state['plat_2_position'].tolist() if hasattr(state['plat_2_position'], 'tolist') else state['plat_2_position']}")
                self._current_step = 4

        elif self._current_step == 4:
            # 4. Pick beaker1
            action, record_array = self.pick_beaker1.forward(
                picking_position=state['beaker_1_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="beaker_l",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                gripper_distances=0.027
            )
            if self.pick_beaker1.is_done():
                print(f"[cleanbeaker] step 4 (pick beaker1) done; beaker1_pos={state['beaker_1_position'].tolist() if hasattr(state['beaker_1_position'], 'tolist') else state['beaker_1_position']}")
                self._current_step = 5

        elif self._current_step == 5:
            # 5. Shake beaker1
            action, record_array = self.shake_beaker1.forward(
                current_joint_positions=self.robot.get_joint_positions(),
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )
            if self.shake_beaker1.is_done():
                print(f"[cleanbeaker] step 5 (shake beaker1) done")
                self._current_step = 6

        elif self._current_step == 6:
            # 6. Pour beaker1 to target_beaker
            action, record_array = self.pour_beaker1.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=state['object_size'],
                source_name="beaker",
                target_position=state['target_position'],
                gripper_position=state['gripper_position'],
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
            )
            if self.pour_beaker1.is_done():
                print(f"[cleanbeaker] step 6 (pour beaker1→target) done; beaker1_pos={state['beaker_1_position'].tolist() if hasattr(state['beaker_1_position'], 'tolist') else state['beaker_1_position']}")
                self._current_step = 7

        elif self._current_step == 7:
            # 7. Place beaker1 to plat1. Uncompensated: with the shallower
            # 10° place yaw its settle shift stays under the 0.04 tolerance
            # (passed consistently pre-fix); see step 3 for the beaker2 story.
            action, record_array = self.place_beaker1.forward(
                place_position=state['plat_1_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_beaker1.is_done():
                print(f"[cleanbeaker] step 7 (place beaker1→plat1) done; beaker1_pos={state['beaker_1_position'].tolist() if hasattr(state['beaker_1_position'], 'tolist') else state['beaker_1_position']} plat1={state['plat_1_position'].tolist() if hasattr(state['plat_1_position'], 'tolist') else state['plat_1_position']}")
                success = self._check_success()
                if success:
                    self._last_failure_reason = ""
                    self.data_collector.write_cached_data(state['joint_positions'][:-1])
                    self._last_success = True
                else:
                    self._last_failure_reason = "Beaker placement check failed: beakers not on target platforms"
                    self.data_collector.clear_cache()
                    self._last_success = False
                done = True
                self.reset_needed = True
                action = None

        if not done and 'camera_data' in state and record_array is not None:
            self.data_collector.cache_step(
                camera_images=state['camera_data'],
                joint_angles=state['joint_positions'][:-1],
                action=record_array,
                language_instruction=self.get_language_instruction()
            )

        return action, done, success
    
    def _step_infer(self, state):
        """
        Executes one step in inference mode.
        Uses policy to process observations and generate actions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        language_instruction = self.get_language_instruction()
        if language_instruction is not None:
            state['language_instruction'] = language_instruction
        else:
            state['language_instruction'] = "Pick up the object from the table"
        
        action = self.inference_engine.step_inference(state)
        
        return action, False, self.is_success()
    
    def _check_success(self):
        # Use world-space geometry centres so the check is independent of
        # how each prim's xform is parented or stacked. The previous code
        # mixed mesh-local "/mesh" xform with parent plat xform, which gave
        # inconsistent z values across episodes.
        beaker1_pos = self.object_utils.get_geometry_center(object_path=self.cfg.beaker_1)
        beaker2_pos = self.object_utils.get_geometry_center(object_path=self.cfg.beaker_2)
        plat1_pos   = self.object_utils.get_geometry_center(object_path=self.cfg.plat_1)
        plat2_pos   = self.object_utils.get_geometry_center(object_path=self.cfg.plat_2)

        if beaker1_pos is None or beaker2_pos is None or plat1_pos is None or plat2_pos is None:
            return False

        def beaker_on_plat(b, p, label):
            dx, dy, dz = abs(b[0] - p[0]), abs(b[1] - p[1]), b[2] - p[2]
            # dx tolerance 0.05 (was 0.04): the side-grasped beaker
            # deterministically settles flat at dx=0.0403 from the EE target
            # (grip-tilt roll on release); the platform is >0.13 m wide, so
            # the beaker is fully on it. 0.04 sat exactly ON the settle
            # point and flipped episodes on sub-millimetre physics noise.
            ok = dx < 0.05 and dy < 0.04 and 0.0 < dz < 0.08
            # Print on success too: the margins tell us how close each place
            # runs to the tolerance (used to calibrate the settle offsets).
            print(
                f"[cleanbeaker debug] {label} {'ok' if ok else 'fail'}: "
                f"dx={dx:.4f} dy={dy:.4f} dz={dz:.4f}"
            )
            return ok

        return beaker_on_plat(beaker1_pos, plat1_pos, "beaker1↔plat1") and beaker_on_plat(beaker2_pos, plat2_pos, "beaker2↔plat2")
    
    def is_success(self):
        Maxframe = 5000
        self.frame_count += 1
        
        if self.frame_count > Maxframe:
            self.reset_needed = True
            return True

        return False

    def get_language_instruction(self) -> Optional[str]:
        step_instructions = {
            1: ('Pick up the second beaker', 'Pick up the second beaker from the table and lift it clear of the surface'),
            2: ('Pour the second beaker into the first beaker', 'Pour the contents of the second beaker into the first beaker carefully'),
            3: ('Place the second beaker on the second platform', 'Move the second beaker to the second platform and set it down carefully'),
            4: ('Pick up the first beaker', 'Pick up the first beaker from the table and lift it clear of the surface'),
            5: ('Shake the first beaker', 'Shake the first beaker to mix the contents thoroughly'),
            6: ('Pour the first beaker into the target beaker', 'Pour the contents of the first beaker into the target beaker carefully'),
            7: ('Place the first beaker on the first platform', 'Move the first beaker to the first platform and set it down carefully'),
        }
        direct, detailed = step_instructions.get(
            self._current_step,
            ('Clean the beakers', 'Complete the current cleaning step carefully'),
        )
        return self._get_cached_instruction(
            f"cleanbeaker:{self._current_step}",
            self._build_instruction_templates(direct, detailed),
        )
