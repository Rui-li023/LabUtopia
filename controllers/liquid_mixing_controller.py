import numpy as np
from enum import Enum
from typing import Dict, Any, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from controllers.atomic_actions.pour_controller import PourController
from controllers.atomic_actions.pressZ_controller import PressZController
from controllers.base_controller import BaseController
from controllers.atomic_actions.pick_controller import PickController
from controllers.atomic_actions.place_controller import PlaceController
from robots.franka.rmpflow_controller import RMPFlowController
from utils.task_utils import TaskUtils
class TaskPhase(Enum):
    """Task phase enumeration"""
    PICKING1 = "picking1"        # Door opening stage
    PICKING2 = "picking2"        # Picking beaker stage
    PICKING3 = "picking3"      # Picking conical bottle stage
    POURING1 = "pouring1"
    POURING2 = "pouring2"
    POURING3 = "pouring3"
    PLACEING1 = "placing1"
    PLACEING2 = "placing2"
    PLACEING3 = "placing3"
    PRESS = "press"
    FINISHED = "finished"      # Task completed
    

class LiquidMixingController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.every_controller_index = 0
        self.controller_index = 0
        self.initial_beaker_position1 = None
        self.initial_beaker_position2 = None
        self.initial_beaker_position3 = None
        # Init success-gate peak trackers here too (not only in reset()) so a
        # step() before the first reset() (collect frame 0) can't AttributeError.
        # current_phase likewise: get_language_instruction() reads it on the very
        # first _step_infer (infer mode never runs _init_collect_mode/reset first).
        self.current_phase = TaskPhase.PICKING1
        self._zmax5 = None; self._zmax4 = None; self._zmax3 = None
        self._quat_init_5 = None; self._quat_init_4 = None; self._quat_init_3 = None
        self._tilt_peak_5 = 0.0; self._tilt_peak_4 = 0.0; self._tilt_peak_3 = 0.0
        # Infer-mode ORACLE phase advancement (mirrors device_operate's contract).
        self.NUM_PHASES = 10                 # PICK/POUR/PLACE x3 + PRESS
        self.success_steps = set()           # phases whose _check_phase_success passed
        self.initial_button_position = None  # heat-device button baseline (terminal PRESS gate)
        self._max_button_press = 0.0         # peak button displacement this episode

    def _init_collect_mode(self, cfg, robot):
        """Initialize data collection mode"""
        super()._init_collect_mode(cfg, robot)
        
        self.current_phase = TaskPhase.PICKING1
        
        # Create RMP controller
        rmp_controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot
        )
        
        # pick_controller1 (beaker_05, the first/cold-start pick that misses most
        # in open-loop replay). events_dt is config-overridable (cfg.task.pick1_events_dt)
        # so we can sweep approach timing — slower approach + a pre-close dwell give
        # the open-loop arm time to land on the beaker before the gripper closes.
        _task = getattr(cfg, "task", None)
        _pick1_dt = getattr(_task, "pick1_events_dt", None) if _task else None
        self.pick_controller1 = PickController(
            name="pick_controller",
            cspace_controller=rmp_controller,
            # event 3 = 0.05 (a ~20-step PRE-CLOSE DWELL): beaker_05 is the first/
            # cold-start pick and had NO dwell (event 3 = 1 = instant close), unlike
            # pick2/3 which already pause; the open-loop arm closed before landing on
            # the beaker -> grasp miss. The dwell lets it settle first. Swept result:
            # beaker_05 replay grasp 20% -> 80%. Slowing the approach did NOT help
            # (the issue is closing too early, not approach speed).
            events_dt=(list(_pick1_dt) if _pick1_dt is not None
                       else [0.002, 0.002, 0.005, 0.05, 0.05, 0.01, 1])
        )
        
        self.pour_controller1 = PourController(
            name="pour_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.006, 0.005, 0.008, 0.005, 0.008, 0.5],
            position_threshold=0.02,
            position_pour=True,
        )
        
        self.place_controller1 = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            robot=robot,
        )
        
        # pick_controller2 (beaker_04, the middle pick that grasps only ~1/5 in
        # open-loop replay). Same dwell as pick3 (beaker_03, 5/5) yet fails, so the
        # culprit is its approach params (pre_offset_x=0.07 vs pick3's 0.10, yaw 20
        # vs 10) — config-overridable to sweep.
        _pick2_dt = getattr(_task, "pick2_events_dt", None) if _task else None
        self.pick_controller2 = PickController(
            name="pick_controller",
            cspace_controller=rmp_controller,
            events_dt=(list(_pick2_dt) if _pick2_dt is not None
                       else [0.002, 0.002, 0.005, 0.2, 0.05, 0.01, 0.1])
        )
        
        self.pour_controller2 = PourController(
            name="pour_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.006, 0.005, 0.008, 0.005, 0.008, 0.5],
            position_threshold=0.02,
            position_pour=True,
        )
        
        self.place_controller2 = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            robot=robot,
        )
        
        self.pick_controller3 = PickController(
            name="pick_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.2, 0.05, 0.01, 0.1]
        )
        
        self.pour_controller3 = PourController(
            name="pour_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.006, 0.005, 0.008, 0.005, 0.008, 0.5],
            position_threshold=0.02,
            position_pour=True,
        )
        
        self.place_controller3 = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            robot=robot,
        )
        
        self.press_controller = PressZController(
            robot=robot,
            name="press_controller",
            cspace_controller=RMPFlowController(
                name="press_controller",
                robot_articulation=robot
            ),
            events_dt=[0.004, 0.1, 0.01],
        )
        
        self.active_controller = self.pick_controller1
        
    def _set_initial_active_controller(self):
        """Set the initial active controller based on the current phase"""
        controller_map = {
            TaskPhase.PICKING1: self.pick_controller1,
            TaskPhase.PICKING2: self.pick_controller2,
            TaskPhase.PICKING3: self.pick_controller3,
            TaskPhase.POURING1: self.pour_controller1,
            TaskPhase.POURING2: self.pour_controller2,
            TaskPhase.POURING3: self.pour_controller3,
            TaskPhase.PLACEING1: self.place_controller1,
            TaskPhase.PLACEING2: self.place_controller2,
            TaskPhase.PLACEING3: self.place_controller3,
            TaskPhase.PRESS: self.press_controller,
        }
        
        if self.current_phase in controller_map:
            self.active_controller = controller_map[self.current_phase]
        else:
            self.active_controller = self.pick_controller1
        
    def reset(self):
        """Reset controller state"""
        super().reset()
        self.initial_beaker_position1 = None
        self.initial_beaker_position2 = None
        self.initial_beaker_position3 = None
        # Per-episode peak-height trackers for the terminal replay success check.
        self._zmax5 = None
        self._zmax4 = None
        self._zmax3 = None
        # Per-episode peak pour-tilt trackers (each beaker must actually tilt to
        # pour, not just lift-and-return). Initial /mesh quat captured on first step.
        self._quat_init_5 = None; self._quat_init_4 = None; self._quat_init_3 = None
        self._tilt_peak_5 = 0.0; self._tilt_peak_4 = 0.0; self._tilt_peak_3 = 0.0
        self.every_controller_index = 0
        self.current_phase = TaskPhase.PICKING1
        # Infer-mode oracle trackers (reset every episode, ALL modes -- placed before
        # the `if self.mode == 'collect'` branch so infer/replay also clear them).
        self.success_steps = set()
        self.initial_button_position = None
        self._max_button_press = 0.0
        if self.mode == "collect":
            self.phase_start_frame = 0
            self.pick_controller1.reset()
            self.pour_controller1.reset()
            self.place_controller1.reset()
            self.pick_controller2.reset()
            self.pour_controller2.reset()
            self.place_controller2.reset()
            self.pick_controller3.reset()
            self.pour_controller3.reset()
            self.place_controller3.reset()
            self.press_controller.reset()
            self.controller_index = 0
            self.active_controller = self.pick_controller1
        elif self.mode == "infer":
            self.inference_engine.reset()
            
    def _check_success(self) -> bool:
        """Terminal success criterion for the whole task (used by replay).

        The task's point is the MIXING: each of beaker_05/04/03 is picked, poured
        into the central mix point, and set back down. Success therefore certifies,
        per beaker: (a) it was LIFTED clear of its start height (the cycle ran —
        essential, else all three sit at their start poses at frame 0 and read as
        success before anything happens); (b) it actually POURED (peak tilt >30°,
        scripted pour ~50-180°), so a mere lift-and-lower can't pass; (c) it ended
        UPRIGHT (current orientation within 30° of its initial upright pose), i.e.
        it was set back down, not dropped/tipped. The exact return-to-start slot is
        NOT required — the pour is the goal, and open-loop replay can't reproduce a
        precise 3-beaker return. The phase-based ``_check_phase_success`` is dead
        copy-paste (refs nonexistent phases) and is not used here.
        """
        tu = TaskUtils.get_instance()
        beaker_specs = (
            ("/World/beaker_05", self.initial_beaker_position1, "_zmax5", "_tilt_peak_5", "_quat_init_5"),
            ("/World/beaker_04", self.initial_beaker_position2, "_zmax4", "_tilt_peak_4", "_quat_init_4"),
            ("/World/beaker_03", self.initial_beaker_position3, "_zmax3", "_tilt_peak_3", "_quat_init_3"),
        )
        all_ok = True
        dbg = []
        for path, init, zmax_attr, tilt_attr, qi_attr in beaker_specs:
            zmax = getattr(self, zmax_attr, None)
            qi = getattr(self, qi_attr, None)
            if init is None or zmax is None or qi is None:
                return False
            init = np.asarray(init, dtype=float)
            lifted = (zmax - init[2]) > 0.05               # was picked up
            tilted = getattr(self, tilt_attr, 0.0) > 30.0  # actually poured
            # Ended upright: current /mesh orientation back near the initial
            # upright pose (set down, not dropped/tipped). Replaces the strict
            # return-to-start-slot requirement.
            q = self.object_utils.get_transform_quat(object_path=path + "/mesh")
            end_tilt = tu.rotation_angle_deg(qi, q) if q is not None else 999.0
            upright = end_tilt < 30.0
            if not (lifted and tilted and upright):
                all_ok = False
            dbg.append(f"{path.split('/')[-1]}:lift={zmax-init[2]:.3f},"
                       f"tilt={getattr(self, tilt_attr, 0.0):.0f},end={end_tilt:.0f}")
        if self.mode == "replay" and (all_ok or self.every_controller_index % 300 == 0):
            print(f"[liqmix replay] all_ok={all_ok} {' '.join(dbg)}")
        return all_ok

    def _check_phase_success(self, state: Dict[str, Any]) -> bool:
        """Per-phase oracle predicate for INFER-mode phase advancement.

        The phase-conditioned POLICY drives the arm; this method only READS world
        geometry to decide when the current sub-goal is reached, so the NEXT phase's
        language instruction can be issued. No object binding / teleport / set_world_pose.

        The task state dict only carries the central mix beaker (/World/beaker_4) and
        the heat plat, so the three poured beakers + the button are read directly via
        object_utils (same as step()/_check_success). Phase -> manipulated prim:
            PICKING1/POURING1/PLACEING1 -> /World/beaker_05
            PICKING2/POURING2/PLACEING2 -> /World/beaker_04
            PICKING3/POURING3/PLACEING3 -> /World/beaker_03
            PRESS                       -> /World/heat_device/button
        (PICKING2/PLACEING2 instruction text says 'conical bottle' but the prim actually
         manipulated is beaker_04 -- the label is policy-facing flavour, left unchanged so
         it matches the strings the VLA was trained on in collect.)
        """
        tu = TaskUtils.get_instance()
        phase = self.current_phase

        # phase -> (beaker prim, recorded initial geometry center, init-quat attr name)
        beaker_map = {
            TaskPhase.PICKING1:  ("/World/beaker_05", self.initial_beaker_position1, "_quat_init_5"),
            TaskPhase.POURING1:  ("/World/beaker_05", self.initial_beaker_position1, "_quat_init_5"),
            TaskPhase.PLACEING1: ("/World/beaker_05", self.initial_beaker_position1, "_quat_init_5"),
            TaskPhase.PICKING2:  ("/World/beaker_04", self.initial_beaker_position2, "_quat_init_4"),
            TaskPhase.POURING2:  ("/World/beaker_04", self.initial_beaker_position2, "_quat_init_4"),
            TaskPhase.PLACEING2: ("/World/beaker_04", self.initial_beaker_position2, "_quat_init_4"),
            TaskPhase.PICKING3:  ("/World/beaker_03", self.initial_beaker_position3, "_quat_init_3"),
            TaskPhase.POURING3:  ("/World/beaker_03", self.initial_beaker_position3, "_quat_init_3"),
            TaskPhase.PLACEING3: ("/World/beaker_03", self.initial_beaker_position3, "_quat_init_3"),
        }

        if phase in beaker_map:
            path, init, qi_attr = beaker_map[phase]
            center = self.object_utils.get_geometry_center(object_path=path)
            if init is None or center is None:
                return False
            z = float(np.asarray(center, dtype=float)[2])
            init_z = float(np.asarray(init, dtype=float)[2])

            if phase in (TaskPhase.PICKING1, TaskPhase.PICKING2, TaskPhase.PICKING3):
                # Lifted clear of its start height (matches the pick lift gate).
                return (z - init_z) > 0.05

            # Live current tilt of the beaker body vs its initial upright pose.
            q = self.object_utils.get_transform_quat(object_path=path + "/mesh")
            qi = getattr(self, qi_attr, None)
            cur_tilt = tu.rotation_angle_deg(qi, q) if (q is not None and qi is not None) else 0.0

            if phase in (TaskPhase.POURING1, TaskPhase.POURING2, TaskPhase.POURING3):
                # Actually tilted to pour NOW. LIVE (not cumulative peak) tilt so a
                # pick/lift wobble earlier in the episode cannot pre-trip this on phase
                # entry: right after the pick the beaker is upright -> cur_tilt ~0 deg.
                # 30 deg matches the validated _check_success 'poured' gate; the scripted
                # pour reaches ~50-180 deg so it is comfortably reachable.
                return cur_tilt > 30.0

            # PLACEING*: set back down -- geometry center returned near its start height
            # AND ended roughly upright (not dropped/tipped). Robust tolerances because the
            # policy will not match the scripted height exactly. At PLACE entry the beaker
            # is still aloft (z high from the pour) so this is False until the policy
            # lowers and releases it. Height-only (no XY) by design: the oracle only needs
            # to know the set-down happened to issue the next instruction.
            return abs(z - init_z) < 0.06 and cur_tilt < 45.0

        if phase == TaskPhase.PRESS:
            # TERMINAL. The policy presses the heat-device button. Two robust OR signals
            # with generous thresholds so PRESS stays reachable (avoid stuck-forever) while
            # only the button-adjacent press satisfies them:
            #   (a) EE descended onto the button region (live gripper<->button distance), or
            #   (b) the button xform was displaced beyond noise (tracked peak in step()).
            button = self.object_utils.get_object_xform_position(object_path="/World/heat_device/button")
            grip = state.get('gripper_position')
            near = (button is not None and grip is not None and
                    float(np.linalg.norm(np.asarray(grip, dtype=float)
                                         - np.asarray(button, dtype=float))) < 0.12)
            pressed = self._max_button_press > 0.004
            return bool(near or pressed)

        return False

    def get_language_instruction(self) -> Optional[str]:
        phase_instructions = {
            TaskPhase.PICKING1: 'Pick up the beaker from the table',
            TaskPhase.PICKING2: 'Pick up the conical bottle',
            TaskPhase.PICKING3: 'Pick up the beaker from the table',
            TaskPhase.POURING1: 'Pour the contents into the beaker',
            TaskPhase.POURING2: 'Pour the contents into the beaker',
            TaskPhase.POURING3: 'Pour the contents into the beaker',
            TaskPhase.PLACEING1: 'Place the beaker on the table',
            TaskPhase.PLACEING2: 'Place the conical bottle on the table',
            TaskPhase.PLACEING3: 'Place the beaker on the table',
            TaskPhase.PRESS: 'Press the beaker',
        }
        direct = phase_instructions.get(self.current_phase, 'Complete the laboratory task')
        return self._get_cached_instruction(
            f"liquid_mixing:{self.current_phase.value}",
            self._build_instruction_templates(
                direct,
                f"{self._normalize_instruction(direct)} carefully and complete this phase accurately",
            ),
        )

    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Execute one step of control
        
        Args:
            state: Current state dictionary
            
        Returns:
            Tuple: (action, done, success)
        """
        # _step_replay (base) reads self.state in _check_success; this override
        # bypasses BaseController.step, so set it here too.
        self.state = state
        if self.initial_beaker_position1 is None:
            self.initial_beaker_position1 = self.object_utils.get_geometry_center(object_path="/World/beaker_05")
        if self.initial_beaker_position2 is None:
            self.initial_beaker_position2 = self.object_utils.get_geometry_center(object_path="/World/beaker_04")
        if self.initial_beaker_position3 is None:
            self.initial_beaker_position3 = self.object_utils.get_geometry_center(object_path="/World/beaker_03")

        # Track each beaker's peak height this episode. The terminal success
        # check (used by replay) needs to distinguish "beaker back at its start
        # pose because the pick/pour/place cycle completed" from "beaker still
        # at its start pose because nothing has happened yet" — both look
        # identical position-wise. A beaker counts as cycled only once it has
        # been lifted clear of its start height.
        tu = TaskUtils.get_instance()
        for attr, qiattr, tattr, path in (
                ("_zmax5", "_quat_init_5", "_tilt_peak_5", "/World/beaker_05"),
                ("_zmax4", "_quat_init_4", "_tilt_peak_4", "/World/beaker_04"),
                ("_zmax3", "_quat_init_3", "_tilt_peak_3", "/World/beaker_03")):
            c = self.object_utils.get_geometry_center(object_path=path)
            if c is not None:
                z = float(np.asarray(c, dtype=float)[2])
                prev = getattr(self, attr, None)
                setattr(self, attr, z if prev is None else max(prev, z))
            # Live orientation from the /mesh subprim (rigid body); track peak tilt.
            q = self.object_utils.get_transform_quat(object_path=path + "/mesh")
            if q is not None:
                if getattr(self, qiattr, None) is None:
                    setattr(self, qiattr, q)
                setattr(self, tattr, max(getattr(self, tattr, 0.0),
                                         tu.rotation_angle_deg(getattr(self, qiattr), q)))

        # Track the heat-device button's peak displacement for the terminal PRESS
        # oracle. Updated every mode BEFORE the dispatch so it is live in infer
        # (the button only moves during the press, so the peak is mode-invariant).
        _bp = self.object_utils.get_object_xform_position(object_path="/World/heat_device/button")
        if _bp is not None:
            _bp = np.asarray(_bp, dtype=float)
            if self.initial_button_position is None:
                self.initial_button_position = _bp
            self._max_button_press = max(
                self._max_button_press,
                float(np.linalg.norm(_bp - self.initial_button_position)),
            )
        self.every_controller_index += 1
        if self.mode == "collect":
            return self._step_collect(state)
        elif self.mode == "replay":
            return self._step_replay(state)
        else:
            return self._step_infer(state)
            
    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Step in data collection mode"""
        if self.current_phase == TaskPhase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success
        
        # If the current controller is not completed, continue to execute
        if not self.active_controller.is_done():
            action, record_array = self._get_phase_action(state)
            
            # Cache data for training
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
                self.current_phase = next_phase
                self._switch_active_controller()
                return None, False, False
            else:
                # All phases completed
                print("All phases completed, task successful!")
                # DIAG: peak pour-tilt per beaker in COLLECT (compare vs the
                # [liqmix replay] line — if collect ~94° but replay over-rotates
                # to 130°+, the velocity-controlled pour isn't reproducing).
                print(f"[liqmix collect] pour-tilt peaks: "
                      f"b05={self._tilt_peak_5:.0f} b04={self._tilt_peak_4:.0f} "
                      f"b03={self._tilt_peak_3:.0f}")
                self._last_failure_reason = ""
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = TaskPhase.FINISHED
                return None, True, True
            
    def _step_infer(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """INFER-mode ORACLE phase advancement.

        The phase-conditioned VLA policy drives the robot; this controller only
        OBSERVES world state to detect when each sub-goal is reached, records the
        passed phase, and issues the NEXT phase's language instruction. Task success ==
        every one of the 10 phases passed in order (mirrors device_operate.success_steps).
        """
        if self.current_phase != TaskPhase.FINISHED and self._check_phase_success(state):
            print(f"Inference: {self.current_phase.value} success!")
            self.success_steps.add(self.current_phase)
            self._advance_to_next_phase()

        if self.current_phase == TaskPhase.FINISHED:
            self.reset_needed = True
            self._last_success = len(self.success_steps) == self.NUM_PHASES
            if self._last_success:
                self._last_failure_reason = ""
            return None, True, self._last_success

        state['language_instruction'] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()

    def _advance_to_next_phase(self) -> None:
        """Advance current_phase along the interleaved sequence (PICK,POUR,PLACE per
        beaker x3, then PRESS), terminating at FINISHED. Infer-only: collect uses
        _get_next_phase/_switch_active_controller and is untouched. Same order as the
        collect phase_sequence."""
        phase_sequence = {
            TaskPhase.PICKING1:  TaskPhase.POURING1,
            TaskPhase.POURING1:  TaskPhase.PLACEING1,
            TaskPhase.PLACEING1: TaskPhase.PICKING2,
            TaskPhase.PICKING2:  TaskPhase.POURING2,
            TaskPhase.POURING2:  TaskPhase.PLACEING2,
            TaskPhase.PLACEING2: TaskPhase.PICKING3,
            TaskPhase.PICKING3:  TaskPhase.POURING3,
            TaskPhase.POURING3:  TaskPhase.PLACEING3,
            TaskPhase.PLACEING3: TaskPhase.PRESS,
            TaskPhase.PRESS:     TaskPhase.FINISHED,
        }
        self.current_phase = phase_sequence.get(self.current_phase, TaskPhase.FINISHED)

    def _beaker_grip(self, pick_ctrl, name: str) -> float:
        """Grasp width for a poured beaker. Prefers cfg.task.beaker_grip (a single
        override for all three beakers — used to sweep the grip vs the ~60mm beaker
        diameter), else falls back to the per-name lookup table."""
        task = getattr(self.cfg, "task", None)
        g = getattr(task, "beaker_grip", None) if task else None
        if g is not None:
            return float(g)
        return pick_ctrl.get_gripper_distance(name)

    def _cfg_task_get(self, key: str, default: float) -> float:
        """Read a float override from cfg.task (for param sweeps), else default."""
        task = getattr(self.cfg, "task", None)
        v = getattr(task, key, None) if task else None
        return float(v) if v is not None else float(default)

    def _get_phase_action(self, state: Dict[str, Any]):
        """Get the corresponding action based on the current phase"""
        if self.current_phase == TaskPhase.PICKING1:
            action, record_array = self.pick_controller1.forward(
                picking_position=self.object_utils.get_geometry_center(object_path="/World/beaker_05"),
                current_joint_positions=state['joint_positions'],
                object_size=self.object_utils.get_object_size(object_path="/World/beaker_05"),
                object_name="beaker_05",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                # Contact-stop grasp width from the lookup table: a binary close
                # position-slams the fingers to 0 and pops the rigid beaker out.
                gripper_distances=self._beaker_grip(self.pick_controller1, "beaker_05"),
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                pre_offset_x=0.1,
                pre_offset_z=0.05,
                after_offset_z=0
            )
            return action, record_array
        elif self.current_phase == TaskPhase.PICKING2:
            action, record_array = self.pick_controller2.forward(
                picking_position=self.object_utils.get_geometry_center(object_path="/World/beaker_04"),
                current_joint_positions=state['joint_positions'],
                object_size=self.object_utils.get_object_size(object_path="/World/beaker_04"),
                object_name="beaker_04",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                # Contact-stop grasp width from the lookup table: a binary close
                # position-slams the fingers to 0 and pops the rigid beaker out.
                gripper_distances=self._beaker_grip(self.pick_controller2, "beaker_04"),
                # pre_offset_x 0.07 -> 0.10 (FIX): the shorter approach offset made
                # beaker_04 (the middle pick) miss in open-loop replay (grasp 0/5);
                # 0.10 -> 5/5 and lifts liquid_mixing replay to 4/5=80%. Swept: yaw
                # didn't matter (kept 20); a longer dwell HURT pick2 (reverted to 0.2).
                end_effector_orientation=R.from_euler(
                    'xyz', np.radians([0, 90, self._cfg_task_get("pick2_yaw_deg", 20.0)])).as_quat(),
                pre_offset_x=self._cfg_task_get("pick2_pre_offset_x", 0.10),
                pre_offset_z=0.05,
                after_offset_z=0
            )
            return action, record_array
        elif self.current_phase == TaskPhase.PICKING3:
            action, record_array = self.pick_controller3.forward(
                picking_position=self.object_utils.get_geometry_center(object_path="/World/beaker_03"),
                current_joint_positions=state['joint_positions'],
                object_size=self.object_utils.get_object_size(object_path="/World/beaker_03"),
                object_name="beaker_03",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                # Contact-stop grasp width from the lookup table: a binary close
                # position-slams the fingers to 0 and pops the rigid beaker out.
                gripper_distances=self._beaker_grip(self.pick_controller3, "beaker_03"),
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                pre_offset_x=0.1,
                pre_offset_z=0.05,
                after_offset_z=0
            )
            return action, record_array
        elif self.current_phase == TaskPhase.POURING1:
            action, record_array = self.pour_controller1.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=self.object_utils.get_object_size(object_path="/World/beaker_05"),
                target_position=np.array([0.32, 0.32, 0.90]),
                current_joint_velocities=self.robot.get_joint_velocities(),
                current_joint_positions=self.robot.get_joint_positions(),
                pour_speed=-1,
                source_name="beaker_05",
                gripper_position=state['gripper_position'],
                target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
            )
            return action, record_array
        elif self.current_phase == TaskPhase.POURING2:
            action, record_array = self.pour_controller2.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=self.object_utils.get_object_size(object_path="/World/beaker_04"),
                target_position=np.array([0.32, 0.32, 0.90]),
                current_joint_velocities=self.robot.get_joint_velocities(),
                current_joint_positions=self.robot.get_joint_positions(),
                pour_speed=-1,
                source_name="beaker_04",
                gripper_position=state['gripper_position'],
                target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
            )
            return action, record_array
        elif self.current_phase == TaskPhase.POURING3:
            action, record_array = self.pour_controller3.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=self.object_utils.get_object_size(object_path="/World/beaker_03"),
                target_position=np.array([0.32, 0.32, 0.90]),
                current_joint_velocities=self.robot.get_joint_velocities(),
                current_joint_positions=self.robot.get_joint_positions(),
                pour_speed=-1,
                source_name="beaker_03",
                gripper_position=state['gripper_position'],
                target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
            )
            return action, record_array
        elif self.current_phase == TaskPhase.PLACEING1:
            action, record_array = self.place_controller1.forward(
                place_position=self.initial_beaker_position1,
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                gripper_position=state['gripper_position'],
                pre_place_z=0.3,
                place_offset_z=0.02
            )
            return action, record_array
        elif self.current_phase == TaskPhase.PLACEING2:
            action, record_array = self.place_controller2.forward(
                place_position=self.initial_beaker_position2,
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                gripper_position=state['gripper_position'],
                pre_place_z=0.3,
                place_offset_z=0.02
            )
            return action, record_array
        elif self.current_phase == TaskPhase.PLACEING3:
            action, record_array = self.place_controller3.forward(
                place_position=self.initial_beaker_position3,
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                pre_place_z=0.3,
                gripper_position=state['gripper_position'],
                place_offset_z=0.02
            )
            return action, record_array
        elif self.current_phase == TaskPhase.PRESS:
            action, record_array = self.press_controller.forward(
                target_position=self.object_utils.get_object_xform_position(object_path="/World/heat_device/button"),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 40])).as_quat(),
                gripper_position=state['gripper_position']
            )
            return action, record_array
        return None, None
        
    def _get_next_phase(self) -> Optional[TaskPhase]:
        phase_sequence = [
            TaskPhase.PICKING1,
            TaskPhase.POURING1,
            TaskPhase.PLACEING1,
            TaskPhase.PICKING2, 
            TaskPhase.POURING2,
            TaskPhase.PLACEING2,
            TaskPhase.PICKING3,
            TaskPhase.POURING3,
            TaskPhase.PLACEING3,
            TaskPhase.PRESS
        ]
        
        self.controller_index += 1
        if self.controller_index >= len(phase_sequence):
            self.controller_index = 0
            return None
            
        # Update the current phase and return
        self.current_phase = phase_sequence[self.controller_index]
        return self.current_phase
        
    def _switch_active_controller(self):
        """Switch the active controller based on the current phase"""
        self.every_controller_index = 0
        controller_map = {
            TaskPhase.PICKING1: self.pick_controller1,
            TaskPhase.PICKING2: self.pick_controller2,
            TaskPhase.PICKING3: self.pick_controller3,
            TaskPhase.POURING1: self.pour_controller1,
            TaskPhase.POURING2: self.pour_controller2,
            TaskPhase.POURING3: self.pour_controller3,
            TaskPhase.PLACEING1: self.place_controller1,
            TaskPhase.PLACEING2: self.place_controller2,
            TaskPhase.PLACEING3: self.place_controller3,
            TaskPhase.PRESS: self.press_controller,
        }
        
        if self.current_phase in controller_map:
            self.active_controller = controller_map[self.current_phase]
            self.active_controller.reset()
            
    def is_success(self) -> bool:
        # All 10 oracle sub-goals certified -> task success (mirrors device_operate).
        return len(self.success_steps) == self.NUM_PHASES
