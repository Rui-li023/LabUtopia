import numpy as np
import random
from enum import Enum
from typing import Dict, Any, Tuple, Optional, List
from scipy.spatial.transform import Rotation as R
from controllers.atomic_actions.pour_controller import PourController
from controllers.base_controller import BaseController
from controllers.atomic_actions.open_controller import OpenController
from controllers.atomic_actions.pick_controller import PickController
from controllers.atomic_actions.place_controller import PlaceController
from robots.franka.rmpflow_controller import RMPFlowController
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from utils.task_utils import TaskUtils

class TaskPhase(Enum):
    OPENING = "opening"
    PICKING1 = "picking1"
    TRANSPORTING = "transporting"
    PICKING2 = "picking2"
    POURING = "pouring"
    TRANSPORTING2 = "transporting2"
    FINISHED = "finished"

class OpenTransportPourController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.every_controller_index = 0
        # Init success-gate peak trackers here too (not only in reset()) so a
        # step() before the first reset() (collect frame 0) can't AttributeError.
        self._z_init_beaker2 = None; self._zmax_beaker2 = None
        self._z_init_conical = None; self._zmax_conical = None
        self._quat_init_conical = None; self._tilt_peak_conical = 0.0

    def _generate_random_sequence(self):
        task_groups = [self.task_group_a, self.task_group_b, self.task_group_c]
        
        random.shuffle(task_groups)
        
        self.randomized_sequence = []
        for group in task_groups:
            self.randomized_sequence.extend(group)
            
        print(f"Generated random task sequence: {[phase.value for phase in self.randomized_sequence]}")
        
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        
        rmp_controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot
        )
        
        self.open_controller = OpenController(
            name="open_controller",
            cspace_controller=rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.0025, 0.005, 0.08, 0.002, 0.05, 0.05, 0.01, 0.008],
            furniture_type="door",
            door_open_direction="clockwise",
            robot=robot,
        )
        
        self.pick_controller1 = PickController(
            name="pick_controller", 
            cspace_controller=rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 1, 0.05, 0.01, 1]
        )
        
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=rmp_controller,
            gripper=robot.gripper,
            robot=robot,
        )
        
        self.pick_controller2 = PickController(
            name="pick_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )
        
        self.pour_controller = PourController(
            name="pour_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.006, 0.005, 0.009, 0.005, 0.009, 0.02]
        )
        
        self.place_controller2 = PlaceController(
            name="place_controller",
            cspace_controller=rmp_controller,
            gripper=robot.gripper,
            robot=robot,
        )
        
        self.task_group_a = [TaskPhase.OPENING]
        self.task_group_b = [TaskPhase.PICKING1, TaskPhase.TRANSPORTING]
        self.task_group_c = [TaskPhase.PICKING2, TaskPhase.POURING, TaskPhase.TRANSPORTING2]
        
        self.randomized_sequence = []
        self._generate_random_sequence()
        self.current_phase = self.randomized_sequence[0]
        
        self._set_initial_active_controller()
        
    def _set_initial_active_controller(self):
        controller_map = {
            TaskPhase.OPENING: self.open_controller,
            TaskPhase.PICKING1: self.pick_controller1,
            TaskPhase.TRANSPORTING: self.place_controller,
            TaskPhase.PICKING2: self.pick_controller2,
            TaskPhase.POURING: self.pour_controller,
            TaskPhase.TRANSPORTING2: self.place_controller2,
        }
        
        if self.current_phase in controller_map:
            self.active_controller = controller_map[self.current_phase]
        else:
            self.active_controller = self.open_controller

    def reset(self):
        super().reset()
        
        self.initial_beaker_position = None
        self.initial_door_position = None
        self.door_opened = False
        self.beaker_picked = False
        self.beaker_transported = False
        self.stir_step_count = 0
        self.every_controller_index = 0
        # Mode-safe peak trackers for the success gates, reset per episode.
        # beaker2 is transported (lift gate); conical_bottle02 is poured (lift +
        # tilt gate). Updated every frame in step() (all modes — phases never
        # advance in replay, so gates must not use the collect-only sub-controllers).
        self._z_init_beaker2 = None; self._zmax_beaker2 = None
        self._z_init_conical = None; self._zmax_conical = None
        self._quat_init_conical = None; self._tilt_peak_conical = 0.0
        if self.mode == "collect":
            self._generate_random_sequence()
            self.current_phase = self.randomized_sequence[0]
            self.phase_start_frame = 0
            self.open_controller.reset()
            self.pick_controller1.reset()
            self.place_controller.reset()
            self.pick_controller2.reset()
            self.pour_controller.reset()
            self.place_controller2.reset()
            self._set_initial_active_controller()
        elif self.mode == "infer":
            self.inference_engine.reset()

    def _track_success_signals(self) -> None:
        """Accumulate peak lift (beaker2 + conical) and peak pour-tilt (conical)
        every frame, all modes. Uses only all-mode signals: live geometry centre
        for height and the /mesh subprim quaternion (where PhysicsRigidBodyAPI
        lives — the top-level prim's xform stays at the authored value) for tilt."""
        tu = TaskUtils.get_instance()
        b2 = self.object_utils.get_geometry_center(object_path="/World/beaker2")
        if b2 is not None:
            z = float(b2[2])
            if self._z_init_beaker2 is None:
                self._z_init_beaker2 = z; self._zmax_beaker2 = z
            self._zmax_beaker2 = max(self._zmax_beaker2, z)
        c = self.object_utils.get_geometry_center(object_path="/World/conical_bottle02")
        if c is not None:
            z = float(c[2])
            if self._z_init_conical is None:
                self._z_init_conical = z; self._zmax_conical = z
            self._zmax_conical = max(self._zmax_conical, z)
        q = self.object_utils.get_transform_quat(object_path="/World/conical_bottle02/mesh")
        if q is not None:
            if self._quat_init_conical is None:
                self._quat_init_conical = q
            self._tilt_peak_conical = max(self._tilt_peak_conical,
                                          tu.rotation_angle_deg(self._quat_init_conical, q))

    def _check_success(self) -> bool:
        """Terminal success criterion for the whole task (used by replay).

        The phase-based ``_check_phase_success`` references phases that don't
        exist in this controller's enum (``PICKING``/``STIRRING``) — dead
        copy-paste that always returns False past phase 1 — and phases never
        advance under replay anyway. Score the task end state directly: beaker2
        ended on ``target_plat`` and conical_bottle02 ended on ``target_plat2``
        (the pour into beaker1 is transient; the bottle is then placed on
        target_plat2). Both placements are absolute positions, so this is true
        only after the episode completes, never at the start. Phase order is
        randomised at collect time, but the final placed state is fixed.
        """
        if not isinstance(self.state, dict):
            return False
        beaker2 = self.object_utils.get_geometry_center(object_path="/World/beaker2")
        plat = self.object_utils.get_geometry_center(object_path="/World/target_plat")
        conical = self.object_utils.get_geometry_center(object_path="/World/conical_bottle02")
        plat2 = self.object_utils.get_geometry_center(object_path="/World/target_plat2")
        if beaker2 is None or plat is None or conical is None or plat2 is None:
            return False
        beaker2 = np.asarray(beaker2, dtype=float)
        plat = np.asarray(plat, dtype=float)
        conical = np.asarray(conical, dtype=float)
        plat2 = np.asarray(plat2, dtype=float)
        beaker_on = (np.linalg.norm(beaker2[:2] - plat[:2]) < 0.06
                     and abs(beaker2[2] - plat[2]) < 0.12)
        conical_on = (np.linalg.norm(conical[:2] - plat2[:2]) < 0.06
                      and abs(conical[2] - plat2[2]) < 0.12)
        if not (beaker_on and conical_on):
            return False

        # Task certification beyond final placement (mode-safe peak trackers):
        #  - beaker2 was lifted (transported), not nudged onto the plat;
        #  - conical was lifted AND actually POURED (tilt >30°) — otp is a pour
        #    task that previously certified zero pours. 30° is calibrated against
        #    the scripted ~50° pour; transport keeps the bottle upright (<30°).
        if self._z_init_beaker2 is None or self._z_init_conical is None:
            return False
        lift_b2 = self._zmax_beaker2 - self._z_init_beaker2
        lift_c = self._zmax_conical - self._z_init_conical
        ok = (lift_b2 > 0.05 and lift_c > 0.05 and self._tilt_peak_conical > 30.0)
        if self.mode == "replay":
            print(f"[otp replay] beaker_on={beaker_on} conical_on={conical_on} "
                  f"lift_b2={lift_b2:.3f} lift_c={lift_c:.3f} "
                  f"tilt_c={self._tilt_peak_conical:.0f} -> {'OK' if ok else 'FAIL'}")
        return ok

    def _check_phase_success(self, state: Dict[str, Any]) -> bool:
        if self.current_phase == TaskPhase.OPENING:
            end_effector_pos = state.get('gripper_position', np.array([0, 0, 0]))
            door_pos = state.get('door_position', np.array([0, 0, 0]))
            distance_to_door = np.linalg.norm(end_effector_pos[:2] - door_pos[:2])
            return distance_to_door < self.DOOR_OPEN_THRESHOLD
            
        elif self.current_phase == TaskPhase.PICKING:
            beaker_pos = state.get('beaker_position', np.array([0, 0, 0]))
            if self.initial_beaker_position is not None:
                height_diff = beaker_pos[2] - self.initial_beaker_position[2]
                return height_diff > self.LIFT_HEIGHT_THRESHOLD
            return False
            
        elif self.current_phase == TaskPhase.TRANSPORTING:
            beaker_pos = state.get('beaker_position', np.array([0, 0, 0]))
            target_pos = state.get('target_position', np.array([0, 0, 0]))
            distance_to_target = np.linalg.norm(beaker_pos[:2] - target_pos[:2])
            height_close = abs(beaker_pos[2] - target_pos[2]) < 0.1
            return distance_to_target < self.TRANSPORT_SUCCESS_THRESHOLD and height_close
            
        elif self.current_phase == TaskPhase.STIRRING:
            self.stir_step_count += 1
            beaker_pos = state.get('beaker_position', np.array([0, 0, 0]))
            stir_tool_pos = state.get('stir_tool_position', np.array([0, 0, 0]))
            
            distance_to_beaker = np.linalg.norm(stir_tool_pos[:2] - beaker_pos[:2])
            in_beaker = distance_to_beaker < 0.05 and stir_tool_pos[2] < beaker_pos[2] + 0.1
            
            return self.stir_step_count > self.STIR_SUCCESS_STEPS and in_beaker
            
        return False
        
    def get_language_instruction(self) -> Optional[str]:
        phase_instructions = {
            TaskPhase.OPENING: 'Open the door of the device',
            TaskPhase.PICKING1: 'Pick up the beaker from the table',
            TaskPhase.TRANSPORTING: 'Transport the beaker to the target platform',
            TaskPhase.PICKING2: 'Pick up the conical bottle',
            TaskPhase.POURING: 'Pour the contents into the beaker',
            TaskPhase.TRANSPORTING2: 'Transport the conical bottle to the target platform',
        }
        direct = phase_instructions.get(self.current_phase, 'Complete the laboratory task')
        return self._get_cached_instruction(
            f"opentransportpour:{self.current_phase.value}",
            self._build_instruction_templates(
                direct,
                f"{self._normalize_instruction(direct)} carefully and complete this phase accurately",
            ),
        )

    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        # _step_replay (base) reads self.state in _check_success; this override
        # bypasses BaseController.step, so set it here too.
        self.state = state
        self._track_success_signals()
        self.every_controller_index += 1
        if self.mode == "collect":
            return self._step_collect(state)
        elif self.mode == "replay":
            return self._step_replay(state)
        else:
            return self._step_infer(state)
            
    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        if self.current_phase == TaskPhase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action, record_array = self._get_phase_action(state)
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=record_array,
                    language_instruction=self.get_language_instruction()
                )
                
            return action, False, False
        else:
            next_phase = self._get_next_phase()
            if next_phase:
                print(f"{self.current_phase.value} phase successful! Switching to {next_phase.value} phase...")
                self.current_phase = next_phase
                self._switch_active_controller()
                return None, False, False
            else:
                print("All phases completed, task successful!")
                self._last_failure_reason = ""
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = TaskPhase.FINISHED
                return None, True, True
            
    def _step_infer(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Step in inference mode"""
        state['language_instruction'] = self.get_language_instruction()
        # Use inference engine to get action
        action = self.inference_engine.step_inference(state)
        
        return action, False, self.is_success()
        
    def _get_phase_action(self, state: Dict[str, Any]):
        """Get corresponding action based on current phase"""
        if self.current_phase == TaskPhase.OPENING:
            action, record_array = self.open_controller.forward(
                handle_position=self.object_utils.get_geometry_center(object_path="/World/MuffleFurnace/handle"),
                revolute_joint_position=self.object_utils.get_revolute_joint_positions(
                    joint_path="/World/MuffleFurnace/RevoluteJoint"
                ),
                current_joint_positions=state['joint_positions'],
                gripper_position=state['gripper_position'],
                end_effector_orientation=euler_angles_to_quats([0, 110, 0], degrees=True, extrinsic=False),
                angle=30,    
            )
            return action, record_array
            
        elif self.current_phase == TaskPhase.PICKING1:
            action, record_array = self.pick_controller1.forward(
                picking_position=self.object_utils.get_geometry_center(object_path="/World/beaker2"),
                current_joint_positions=state['joint_positions'],
                object_size=self.object_utils.get_object_size(object_path="/World/beaker2"),
                object_name="beaker2",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 60])).as_quat(),
                pre_offset_x=0.1,
                pre_offset_z=0.05,
                after_offset_z=0.05
            )
            return action, record_array
        elif self.current_phase == TaskPhase.PICKING2:
            action, record_array = self.pick_controller2.forward(
                picking_position=self.object_utils.get_geometry_center(object_path="/World/conical_bottle02"),
                current_joint_positions=state['joint_positions'],
                object_size=self.object_utils.get_object_size(object_path="/World/conical_bottle02"),
                object_name="conical_bottle02",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                # 0.018: the peak of a shallow curve — 0.018→7/10, 0.016→5/10,
                # 0.022→4/10. The tapered conical slips under the open-loop pour
                # torque; firmer doesn't help past 0.018 and looser is worse.
                # ~70% is the grip-distance ceiling for this grasp.
                gripper_distances=0.018,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                pre_offset_x=0.07,
                pre_offset_z=0.05,
                after_offset_z=0.05
            )
            return action, record_array
        elif self.current_phase == TaskPhase.TRANSPORTING:
            action, record_array = self.place_controller.forward(
                place_position=self.object_utils.get_geometry_center(object_path="/World/target_plat"),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 60])).as_quat(),
                gripper_position=state['gripper_position']
            )
            return action, record_array
        elif self.current_phase == TaskPhase.POURING:
            action, record_array = self.pour_controller.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=self.object_utils.get_object_size(object_path="/World/conical_bottle02"),
                target_position=self.object_utils.get_geometry_center(object_path="/World/beaker1"),
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
                source_name="conical_bottle02",
                gripper_position=state['gripper_position'],
                target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
            )
            return action, record_array
        elif self.current_phase == TaskPhase.TRANSPORTING2:
            action, record_array = self.place_controller2.forward(
                place_position=self.object_utils.get_geometry_center(object_path="/World/target_plat2"),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                gripper_position=state['gripper_position'],
                place_offset_z=0.115
            )
            return action, record_array
        return None, None
        
    def _get_next_phase(self) -> Optional[TaskPhase]:
        """Get next phase (based on random sequence)"""
        try:
            current_idx = self.randomized_sequence.index(self.current_phase)
            if current_idx < len(self.randomized_sequence) - 1:
                return self.randomized_sequence[current_idx + 1]
        except ValueError:
            pass
        return None
        
    def _switch_active_controller(self):
        """Switch active controller based on current phase"""
        self.every_controller_index = 0
        controller_map = {
            TaskPhase.OPENING: self.open_controller,
            TaskPhase.PICKING1: self.pick_controller1,
            TaskPhase.TRANSPORTING: self.place_controller,
            TaskPhase.PICKING2: self.pick_controller2,
            TaskPhase.POURING: self.pour_controller,
            TaskPhase.TRANSPORTING2: self.place_controller2,
            TaskPhase.FINISHED: self.open_controller,
        }
        
        if self.current_phase in controller_map:
            self.active_controller = controller_map[self.current_phase]
            self.active_controller.reset()
            
    def is_success(self) -> bool:
        return False