import numpy as np
import random
from enum import Enum
from typing import Dict, Any, Tuple, Optional
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
        # current_phase likewise: get_language_instruction() reads it on the very
        # first _step_infer (infer mode never runs _init_collect_mode/reset first).
        self.current_phase = TaskPhase.OPENING
        self._z_init_beaker2 = None; self._zmax_beaker2 = None
        self._z_init_conical = None; self._zmax_conical = None
        self._quat_init_conical = None; self._tilt_peak_conical = 0.0
        # --- infer-mode oracle phase advancement (mirror DeviceOperateController) ---
        # Defined here (not only in _init_collect_mode, which runs in COLLECT mode
        # only) so the infer-mode oracle has a phase sequence to follow and never
        # AttributeErrors on a _step_infer before the first reset(). The hasattr
        # guards leave the collect-time shuffle from _init_collect_mode untouched
        # (in collect those attrs already exist; in infer they do not).
        self.success_steps = set()
        self.N_PHASES = 6  # OPENING, PICKING1, TRANSPORTING, PICKING2, POURING, TRANSPORTING2
        if not hasattr(self, "task_group_a"):
            self.task_group_a = [TaskPhase.OPENING]
            self.task_group_b = [TaskPhase.PICKING1, TaskPhase.TRANSPORTING]
            self.task_group_c = [TaskPhase.PICKING2, TaskPhase.POURING, TaskPhase.TRANSPORTING2]
        if not hasattr(self, "randomized_sequence"):
            self.randomized_sequence = list(self.task_group_a + self.task_group_b + self.task_group_c)
        # Frame-0 baseline of the muffle-furnace door handle for the OPENING gate.
        # CURRENT-displacement (not a peak): the door is closed until OPENING and
        # stays open once pulled, so a live displacement can't latch on a transient
        # bump in an earlier phase (OPENING may be scheduled last).
        self._handle_init = None

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
        
        # Position-controlled pour (see pick_pour_controller for the full
        # rationale): the velocity pour recorded the integrated commanded
        # velocity as the wrist action, which sat 13-24 deg BELOW the measured
        # state across the whole return leg, so a closed-loop policy is never
        # shown a "raise the wrist" command. Position pour records the command
        # it actually sends and parks upright at the end.
        self.pour_controller = PourController(
            name="pour_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.006, 0.005, 0.009, 0.005, 0.009, 0.02],
            position_pour=bool(getattr(getattr(cfg, "task", None), "position_pour", True)),
            pour_angle_rad=float(getattr(getattr(cfg, "task", None), "pour_angle_rad", 1.2)),
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
        self.success_steps.clear()
        self._handle_init = None
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
            self._generate_random_sequence()
            self.current_phase = self.randomized_sequence[0]

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
        # Frame-0 baseline of the muffle-furnace door handle for the OPENING
        # gate (current-displacement check in _check_phase_success). Captured
        # once, every mode, post-warmup; the handle is static until pulled.
        h = self.object_utils.get_geometry_center(object_path="/World/MuffleFurnace/handle")
        if h is not None and self._handle_init is None:
            self._handle_init = np.asarray(h, dtype=float)

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
        """Per-phase world-state predicate for the infer-mode oracle.

        Reads object geometry DIRECTLY from the stage via object_utils: the otp
        task state dict only carries beaker2 ('object_position') and target_plat
        ('target_position'); the door handle, conical bottle and second platform
        are NOT in the state dict (the old code's state.get('door_position') /
        'beaker_position' / 'stir_tool_position' silently defaulted to [0,0,0] and
        branched on PICKING / STIRRING phases that don't exist in this enum). All
        baselines come from trackers updated every frame in _track_success_signals()
        (called before the mode dispatch), so they are live under the policy. Only
        the CURRENT phase is ever evaluated, so an earlier phase cannot pre-trip a
        later phase's predicate.
        """
        tu = TaskUtils.get_instance()
        if self.current_phase == TaskPhase.OPENING:
            # Door pulled open: handle swung away from its frame-0 (closed)
            # position. CURRENT displacement, not a peak: the door is closed
            # until OPENING and stays open once pulled, so a transient bump in
            # an earlier phase can't latch this gate. ~0.04 m matches the
            # scripted 30 deg open (device_operate uses 0.13 m for ~100 deg).
            h = self.object_utils.get_geometry_center(object_path="/World/MuffleFurnace/handle")
            if h is None or self._handle_init is None:
                return False
            return float(np.linalg.norm(np.asarray(h, dtype=float) - self._handle_init)) > 0.04
        elif self.current_phase == TaskPhase.PICKING1:
            b2 = self.object_utils.get_geometry_center(object_path="/World/beaker2")
            if b2 is None or self._z_init_beaker2 is None:
                return False
            return float(b2[2]) - self._z_init_beaker2 > 0.05
        elif self.current_phase == TaskPhase.TRANSPORTING:
            b2 = self.object_utils.get_geometry_center(object_path="/World/beaker2")
            plat = self.object_utils.get_geometry_center(object_path="/World/target_plat")
            if b2 is None or plat is None:
                return False
            b2 = np.asarray(b2, dtype=float)
            plat = np.asarray(plat, dtype=float)
            return (np.linalg.norm(b2[:2] - plat[:2]) < 0.06
                    and abs(b2[2] - plat[2]) < 0.12)
        elif self.current_phase == TaskPhase.PICKING2:
            c = self.object_utils.get_geometry_center(object_path="/World/conical_bottle02")
            if c is None or self._z_init_conical is None:
                return False
            return float(c[2]) - self._z_init_conical > 0.05
        elif self.current_phase == TaskPhase.POURING:
            # CURRENT tilt (not the persistent peak) so a PICKING2 grasp wobble
            # can't pre-satisfy the pour; scripted pour ~50 deg, grasp << 30 deg.
            q = self.object_utils.get_transform_quat(object_path="/World/conical_bottle02/mesh")
            if q is None or self._quat_init_conical is None:
                return False
            return tu.rotation_angle_deg(self._quat_init_conical, q) > 30.0
        elif self.current_phase == TaskPhase.TRANSPORTING2:
            c = self.object_utils.get_geometry_center(object_path="/World/conical_bottle02")
            plat2 = self.object_utils.get_geometry_center(object_path="/World/target_plat2")
            if c is None or plat2 is None:
                return False
            c = np.asarray(c, dtype=float)
            plat2 = np.asarray(plat2, dtype=float)
            return (np.linalg.norm(c[:2] - plat2[:2]) < 0.06
                    and abs(c[2] - plat2[2]) < 0.12)
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
        """Infer-mode oracle phase advancement (mirror DeviceOperateController).

        The POLICY drives the robot; this only OBSERVES world state to decide
        when the current sub-goal is reached, records it in success_steps, and
        then issues the NEXT phase's language instruction. Phase order follows
        the group-randomised self.randomized_sequence via _get_next_phase(), so
        it is NOT hard-coded. Success = all N_PHASES detected in order.

        Does NOT call _advance_to_next_phase()/_switch_active_controller(): those
        reset the atomic sub-controllers (open_controller, ...) which are created
        only in _init_collect_mode and DO NOT EXIST in infer mode.
        """
        if self.current_phase != TaskPhase.FINISHED and self._check_phase_success(state):
            print(f"Inference: {self.current_phase.value} success!")
            self.success_steps.add(self.current_phase)
            next_phase = self._get_next_phase()
            self.current_phase = next_phase if next_phase is not None else TaskPhase.FINISHED

        if self.current_phase == TaskPhase.FINISHED:
            self.reset_needed = True
            self._last_success = len(self.success_steps) == self.N_PHASES
            if self._last_success:
                self._last_failure_reason = ""
            return None, True, self._last_success

        state['language_instruction'] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        return action, False, len(self.success_steps) == self.N_PHASES

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
                current_joint_positions=self.robot.get_joint_positions(),
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
        return len(self.success_steps) == self.N_PHASES
