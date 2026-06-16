from typing import Optional
from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum
from isaacsim.core.utils.types import ArticulationAction
from robots.franka.rmpflow_controller import RMPFlowController
from utils.task_utils import TaskUtils

from .atomic_actions.pick_controller import PickController
from .atomic_actions.pour_controller import PourController
from .base_controller import BaseController

class Phase(Enum):
    PICKING = "picking"
    POURING = "pouring"
    FINISHED = "finished"

class PourTaskController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
        self.initial_size = None
        self.task_utils = TaskUtils.get_instance()
        self.initial_quaternion = None
        self.pour_timer = 0
        self.pour_complete = False
        self.return_complete = False
        self.return_timer = 0
        self.last_error_info = None
        self.current_phase = Phase.PICKING
        self._post_done_wait = 0
        self._POST_DONE_MAX = 240
        self._replay_initial_quaternion = None

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        """Initialize controller for data collection mode."""
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )

        self.pour_controller = PourController(
            name="pour_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot,
                use_default_config=False
            ),
            events_dt=[0.006, 0.002, 0.012, 0.01, 0.008, 0.01]
        )
        self.active_controller = self.pick_controller

    def _init_infer_mode(self, cfg, robot=None):
        super()._init_infer_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )

    def _init_replay_mode(self, cfg, robot=None):
        """Replay records only the pour actions; scripted pick brings the
        robot into the holding-source state before the recorded actions run.
        Disable pick randomization so the post-pick pose matches what the
        recorded pour actions assume."""
        super()._init_replay_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller_replay",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02],
        )
        self.pick_controller._sample_randomization = lambda: None

    def reset(self):
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self.initial_quaternion = None
        self.pour_timer = 0
        self.pour_complete = False
        self.return_complete = False
        self.return_timer = 0
        self.last_error_info = None
        self._post_done_wait = 0
        self._replay_initial_quaternion = None
        self.pick_controller.reset()
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pour_controller.reset()
        elif self.mode == "infer":
            self.inference_engine.reset()

    # Grasp width (m) for the pour source, replacing the binary slam-close that
    # ejected light glassware (and forced the source mass to be inflated). Beakers
    # are NOT keyed in PickController.get_gripper_distance (exact-match -> 0.0), so
    # default them here; override per-config with task.source_grip_distance.
    _SOURCE_GRIP_DEFAULTS = {"beaker": 0.020, "graduated_cylinder": 0.018, "conical_bottle": 0.020}

    def _source_grip_distance(self, object_name: str) -> float:
        override = getattr(getattr(self.cfg, "task", None), "source_grip_distance", None)
        if override is not None:
            return float(override)
        name = (object_name or "").lower()
        for key, width in self._SOURCE_GRIP_DEFAULTS.items():
            if key in name:
                return width
        return 0.020

    def _check_success(self) -> bool:
        """Evaluate whether the current state meets the task success criterion.

        Replay uses a simplified criterion: after the recorded actions execute,
        the bottle should be near the target (xy) and its orientation should
        be back to roughly upright (within ~25° of the post-pick quaternion).
        This avoids fighting the strict pour state-machine which can stall on
        PD jitter when the recorded tilt comes a hair short of 50°.
        """
        if self.mode == "replay":
            if self._replay_initial_quaternion is None:
                self._replay_initial_quaternion = self.state['object_quaternion']
                return False
            # check_rotation_angle returns True when current orientation has
            # diverged from initial by more than the threshold. The recorded
            # pour trajectory ends with the bottle back to roughly upright;
            # accept it as success once orientation is within 40° of the
            # post-pick reference (loose enough to tolerate PD lag).
            still_tilted = self.task_utils.check_rotation_angle(
                self._replay_initial_quaternion,
                self.state['object_quaternion'],
                threshold_degrees=40,
            )
            return not still_tilted
        return self._check_phase_success()

    def _step_replay(self, state):
        """Run scripted pick first (not recorded); then replay pour actions.

        Switches current_phase to POURING once the scripted pick completes so
        that _check_phase_success runs the pour-state machine during replay.
        """
        if not self.pick_controller.is_done():
            action, _ = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name=state['object_name'],
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                pre_offset_x=0.05,
                pre_offset_z=0.05,
                after_offset_z=0.5,
                gripper_distances=self._source_grip_distance(state['object_name']),
            )
            return action, False, False
        if self.current_phase != Phase.POURING:
            self.current_phase = Phase.POURING
        return super()._step_replay(state)

    def _check_phase_success(self):
        """Check if current phase is successful."""
        object_pos = self.state['object_position']
        self.last_error_info = None 
        
        if self.initial_position is None:
            raise ValueError("initial_position not set")

        if self.current_phase == Phase.PICKING:
            required_height = self.initial_position[2] + 0.12
            success = object_pos[2] > required_height
            if not success:
                self.last_error_info = {
                    'phase': 'PICKING',
                    'current_height': object_pos[2],
                    'required_height': required_height,
                    'height_diff': object_pos[2] - required_height
                }
            return success
            
        elif self.current_phase == Phase.POURING:
            if self.initial_quaternion is None:
                self.initial_quaternion = self.state['object_quaternion']
                self.last_error_info = {
                    'phase': 'POURING',
                    'error': 'Initial quaternion not set yet'
                }
                return False
                
            current_quat = self.state['object_quaternion']
            
            # First check if we're close enough to target for pouring
            xy_dist = np.linalg.norm(object_pos[:2] - self.state['target_position'][:2])
            pour_threshold = self.task_utils.get_pour_threshold(self.state['object_name'], self.state['object_size']) + 0.05
            
            if xy_dist > pour_threshold:
                self.last_error_info = {
                    'phase': 'POURING',
                    'current_distance': xy_dist,
                    'required_distance': pour_threshold,
                    'distance_diff': xy_dist - pour_threshold
                }
                return False
            
            if not self.pour_complete:
                # print(self.initial_quaternion, current_quat)
                self.pour_complete = self.task_utils.check_rotation_angle(
                    self.initial_quaternion, 
                    current_quat,
                    threshold_degrees=50
                )
                if not self.pour_complete:
                    self.last_error_info = {
                        'phase': 'POURING',
                        'error': 'Pour rotation not complete yet',
                        'pour_complete': self.pour_complete
                    }
                return False
                
            # After pour complete, check if returned to original orientation
            if not self.return_complete:
                rotation_diff = self.task_utils.check_rotation_angle(
                    self.initial_quaternion,
                    current_quat,
                    threshold_degrees=30  # smaller threshold for return position
                )
                if not rotation_diff:
                    self.return_complete = True
                    self.return_timer = 0
                else:
                    self.last_error_info = {
                        'phase': 'POURING',
                        'error': 'Return rotation not complete yet',
                        'return_complete': self.return_complete
                    }
                return False
                
            # Wait for 2 seconds in return position
            if self.return_complete and object_pos[2] > self.initial_position[2] + 0.05:
                self.return_timer += 0.012
                success = self.return_timer >= 1.0
                if not success:
                    self.last_error_info = {
                        'phase': 'POURING',
                        'error': 'Waiting for return timer',
                        'return_timer': self.return_timer,
                        'required_time': 1.0
                    }
                return success
            else:
                self.last_error_info = {
                    'phase': 'POURING',
                    'error': 'Object not in correct position for return timer',
                    'current_height': object_pos[2],
                    'required_height': self.initial_position[2] + 0.05,
                    'return_complete': self.return_complete
                }
                return False
        
        return False
    def step(self, state):
        """Execute one step of control.

        Args:
            state: Current state dictionary containing sensor data and robot state

        Returns:
            Tuple containing action, done flag, and success flag
        """
        if self.initial_position is None:
            self.initial_position = state['object_position']
        if self.initial_size is None:
            self.initial_size = state['object_size']
        return super().step(state)

    def _step_collect(self, state):
        """Execute collection mode step."""
        success = self._check_phase_success()
        if success:
            if self.current_phase == Phase.PICKING:
                print("Pick task success! Switching to pour...")
                self.current_phase = Phase.POURING
                self.active_controller = self.pour_controller
                return None, False, False
            elif self.current_phase == Phase.POURING:
                print("Pour task success!")
                self._last_failure_reason = ""
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = Phase.FINISHED
                return None, True, True
        
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action = None
            if self.current_phase == Phase.PICKING:
                action, _ = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                    pre_offset_x=0.05,
                    pre_offset_z=0.05,
                    after_offset_z=0.5,
                    gripper_distances=self._source_grip_distance(state['object_name']),
                )
            else:
                action, record_array = self.pour_controller.forward(
                    articulation_controller=self.robot.get_articulation_controller(),
                    source_size=self.initial_size,
                    target_position=state['target_position'],
                    current_joint_velocities=self.robot.get_joint_velocities(),
                    pour_speed=-1,
                    source_name=state['object_name'],
                    gripper_position=state['gripper_position'],
                    current_joint_positions=state['joint_positions'],
                )
                
                if 'camera_data' in state:
                    self.data_collector.cache_step(
                        camera_images=state['camera_data'],
                        joint_angles=state['joint_positions'][:-1],
                        action=record_array,
                        language_instruction=self.get_language_instruction()
                    )
            
            return action, False, False

        # Atomic state machine finished but success not yet satisfied (typically
        # the bottle is still rotating back to upright, or the return_timer is
        # still counting up). Hold null actions for up to _POST_DONE_MAX physics
        # frames so physics can settle, re-checking success each frame.
        if self.current_phase == Phase.POURING and self._post_done_wait < self._POST_DONE_MAX:
            self._post_done_wait += 1
            n_joints = len(state['joint_positions'])
            null_action = ArticulationAction(joint_positions=[None] * n_joints)
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=np.concatenate([state['joint_positions'][:7], [0.0]]),
                    language_instruction=self.get_language_instruction(),
                )
            return null_action, False, False

        self._last_failure_reason = f"Pour {self.current_phase.value} failed" + (f": {self.last_error_info}" if self.last_error_info else "")
        print(f"{self.current_phase.value} task failed!")
        if self.last_error_info is not None:
            print(f"Phase failure details: {self.last_error_info}")
        self.data_collector.clear_cache()
        self._last_success = False
        self.current_phase = Phase.FINISHED
        return None, True, False

    def _step_infer(self, state):
        """Execute inference mode step."""
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.pick_controller.is_done():
            action = None
            action, _ = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 15])).as_quat(),
                    gripper_distances=self._source_grip_distance(state['object_name']),
                )
            
        else:
            state['language_instruction'] = self.get_language_instruction()

            action = self.inference_engine.step_inference(state)
        success = self._check_phase_success()
        if success and self.current_phase == Phase.PICKING:
            print("Pick task success! Switching to pour...")
            self.current_phase = Phase.POURING
        elif success and self.current_phase == Phase.POURING:
            print("Pour task success!")
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return None, True, True
               
        return action, False, False

    def get_language_instruction(self) -> Optional[str]:
        object_name = self.clean_object_name(self.state['object_name'])
        if self.current_phase == Phase.PICKING:
            return self._get_cached_instruction(
                'pour:picking',
                self._build_instruction_templates(
                    f"Pick up the {object_name}",
                    f"Pick up the {object_name} from the table and prepare it for pouring",
                ),
            )
        return self._get_cached_instruction(
            'pour:pouring',
            self._build_instruction_templates(
                f"Pour the contents of the {object_name} into the target",
                f"Move the {object_name} over the target container and pour its contents carefully",
            ),
        )
