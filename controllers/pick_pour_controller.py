from enum import Enum
from typing import Any, Optional

import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from loguru import logger

from utils.task_utils import TaskUtils
from .atomic_actions.pick_controller import PickController
from .atomic_actions.pour_controller import PourController
from .base_controller import BaseController
from .grasp_frame import GraspFrame, resolve_pick_z_offset

class Phase(Enum):
    PICKING = "picking"
    POURING = "pouring"
    FINISHED = "finished"

class PickPourTaskController(BaseController):
    def __init__(self, cfg, robot):
        """Initialize the pick and pour task controller.
        
        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
        """
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
        # Which language instruction the POURING phase gets at inference time.
        # "pour" (default) keeps the historical behaviour; "pick" reuses the
        # PICKING sentence, which is what the exported LeRobot episodes are
        # actually labelled with (the exporter stamps every frame with the first
        # frame's task_index), so the policy is queried in-vocabulary.
        self._pour_prompt_mode = "pour"
        infer_cfg = getattr(cfg, "infer", None)
        if infer_cfg is not None and "pour_phase_prompt" in infer_cfg:
            self._pour_prompt_mode = str(infer_cfg.pour_phase_prompt)
        self._reset_gate_diag()

    # Ordered legs of the pour gate (see _check_phase_success). Tracking the
    # deepest leg reached is the only way to tell "never lifted" from "poured
    # but never returned" — the infer path used to print nothing at all.
    GATE_STAGES = (
        "0:not_lifted",
        "1:lifted_xy_far",
        "2:xy_ok_tilt_pending",
        "3:tilt_ok_return_pending",
        "4:returned_hold_pending",
        "5:success",
    )

    def _fmt_swing(self) -> str:
        """Swing envelope, or n/a for episodes that never entered POURING."""
        if self._gate_j7_min == float("inf"):
            return "n/a"
        return f"[{self._gate_j7_min:.1f},{self._gate_j7_max:.1f}]deg"

    def _reset_gate_diag(self) -> None:
        self._gate_stage_best = -1
        self._gate_peak_tilt = 0.0
        self._gate_min_tilt_after_peak = float("inf")
        self._gate_min_xy = float("inf")
        # Signed wrist-roll (j7) swing relative to the pour-phase start. The
        # quaternion gate only sees |angle|, which cannot distinguish "poured
        # 45 deg" from "over-rotated 45 deg the other way" — the demos do both
        # (tilt down, swing back through upright, keep going).
        self._gate_j7_ref = None
        self._gate_j7_min = float("inf")
        self._gate_j7_max = float("-inf")

    def _init_collect_mode(self, cfg, robot):
        """Initialize controller for data collection mode."""
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )
        
        # Position-controlled pour tilt (task.position_pour, default on). The old
        # velocity pour recorded the *integrated commanded velocity* as the wrist
        # angle, which has no causal link to the executed joint: across the whole
        # return leg the recorded action sat 13-24 deg BELOW the measured state in
        # 100/100 demos, so a state-conditioned policy is never shown a "raise the
        # wrist" command and stalls inverted. Position pour records the command it
        # actually sends, and ends by ramping back to the pour-start angle and
        # holding there — which also gives the demos the static upright terminal
        # segment the velocity pour never had.
        # pour_angle_rad 1.2 (69 deg) matches the tilt the velocity pour actually
        # executed (j7 swing -53..-77 deg); the atomic default 2.0 (115 deg) would
        # be a much larger motion than any existing demo.
        task_cfg = getattr(cfg, "task", None)
        self.pour_controller = PourController(
            name="pour_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.006, 0.002, 0.012, 0.01, 0.008, 0.01],
            position_pour=bool(getattr(task_cfg, "position_pour", True)),
            pour_angle_rad=float(getattr(task_cfg, "pour_angle_rad", 1.2)),
        )

        # Both phases used to command a hard-coded world-frame wrist pose, correct
        # only for a source beaker and a receiving beaker straight ahead of the
        # base. With `grasp.bearing_gain: 1.0` each frame follows the bearing of
        # what it is reaching for, which is what lets both spawn ranges widen.
        # Defaults reproduce the historical fixed poses exactly.
        grasp_cfg = getattr(cfg, "grasp", None)
        self._pick_frame = GraspFrame(grasp_cfg, robot, (0.0, 90.0, 30.0), label="grasp/pick")
        self._pour_frame = GraspFrame(
            grasp_cfg, robot, (0.0, 90.0, 15.0), prefix="pour_", label="grasp/pour"
        )
        pick_z_offset = resolve_pick_z_offset(grasp_cfg)
        if pick_z_offset is not None:
            self.pick_controller.pick_z_offset_override = pick_z_offset
            logger.info(f"[grasp] pick_z_offset override = {pick_z_offset:.3f} m above the object origin")

        self.active_controller = self.pick_controller

    def reset(self):
        """Reset controller state and phase."""
        if getattr(self, "mode", "") != "collect" and self._gate_stage_best >= 0:
            logger.info(
                f"[POUR-GATE-SUMMARY] deepest={self.GATE_STAGES[self._gate_stage_best]} "
                f"peak_tilt={self._gate_peak_tilt:.1f}deg "
                f"min_tilt_after_peak={self._gate_min_tilt_after_peak:.1f}deg "
                f"min_xy={self._gate_min_xy:.3f}m "
                f"j7_swing={self._fmt_swing()} "
                f"pour_done={self.pour_complete} return_done={self.return_complete} "
                f"hold_timer={self.return_timer:.2f}s "
                f"pour_prompt={self._pour_prompt_mode}"
            )
        self._reset_gate_diag()
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
        
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pick_controller.reset()
            self.pour_controller.reset()
            self._pick_frame.new_episode()
            self._pour_frame.new_episode()
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
        """Evaluate whether the current state meets the task success criterion."""
        return self._check_phase_success()

    def _check_phase_success(self):
        """Evaluate the gate, and in infer/replay also record the stage diagnostics.

        Wrapping here (rather than calling from ``_step_infer``) means replay —
        which reaches the gate through ``BaseController._step_replay`` — gets the
        same ``[POUR-GATE]`` trace, so expert demos and policy rollouts are
        measured with one instrument.
        """
        ok = self._evaluate_phase_gate()
        if getattr(self, "mode", "") != "collect" and self.state is not None:
            self._update_gate_diag(self.state, ok)
        return ok

    def _evaluate_phase_gate(self):
        """Check if current phase is successful."""
        object_pos = self.state['object_position']
        self.last_error_info = None 
        
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

    def _gate_stage(self, success: bool) -> int:
        """Map the current gate rejection into an ordered stage index.

        ``success`` is *phase* success, not task success: in PICKING it only
        means the 0.12 m lift passed, which is the entry into the pour legs.
        """
        if success:
            return 5 if self.current_phase == Phase.POURING else 1
        if self.current_phase == Phase.PICKING:
            return 0
        info: dict[str, Any] = self.last_error_info or {}
        error = str(info.get("error", ""))
        if "current_distance" in info:
            return 1
        if error.startswith("Pour rotation"):
            return 2
        if error.startswith("Return rotation"):
            return 3
        if error.startswith("Waiting for return timer") or error.startswith("Object not in correct"):
            return 4
        return 1

    def _update_gate_diag(self, state, success: bool) -> None:
        """Track how deep into the pour gate the episode got, and log advances.

        Only ``_step_collect`` ever printed ``last_error_info``; inference runs
        emitted nothing, so a failed episode gave no clue which leg of the gate
        (xy alignment / 50 deg tilt / return / hold) rejected it. Also keeps
        ``_last_failure_reason`` populated so ``print_failure_reason`` reports
        something on the timeout path.
        """
        stage = self._gate_stage(success)

        if self.initial_quaternion is not None and state.get("object_quaternion") is not None:
            tilt = float(self.task_utils.rotation_angle_deg(
                self.initial_quaternion, state["object_quaternion"]))
            self._gate_peak_tilt = max(self._gate_peak_tilt, tilt)
            # The return leg needs the tilt back under 30 deg AFTER the 50 deg
            # peak; tracking the post-peak minimum says how close it got.
            if self.pour_complete:
                self._gate_min_tilt_after_peak = min(self._gate_min_tilt_after_peak, tilt)
        if self.current_phase == Phase.POURING and state.get("joint_positions") is not None:
            j7 = float(np.asarray(state["joint_positions"])[6])
            if self._gate_j7_ref is None:
                self._gate_j7_ref = j7
            dj7 = float(np.degrees(j7 - self._gate_j7_ref))
            self._gate_j7_min = min(self._gate_j7_min, dj7)
            self._gate_j7_max = max(self._gate_j7_max, dj7)
        if (self.current_phase == Phase.POURING
                and state.get("target_position") is not None):
            xy = float(np.linalg.norm(
                np.asarray(state["object_position"])[:2]
                - np.asarray(state["target_position"])[:2]))
            self._gate_min_xy = min(self._gate_min_xy, xy)

        if stage > self._gate_stage_best:
            self._gate_stage_best = stage
            logger.info(
                f"[POUR-GATE] -> {self.GATE_STAGES[stage]} "
                f"peak_tilt={self._gate_peak_tilt:.1f}deg "
                f"min_tilt_after_peak={self._gate_min_tilt_after_peak:.1f}deg "
                f"min_xy={self._gate_min_xy:.3f}m "
                f"detail={self.last_error_info}"
            )

        if success:
            self._last_failure_reason = ""
        else:
            self._last_failure_reason = (
                f"pour gate stopped at {self.GATE_STAGES[self._gate_stage_best]} "
                f"(peak_tilt={self._gate_peak_tilt:.1f}deg, min_xy={self._gate_min_xy:.3f}m)"
            )

    def step(self, state):
        """Execute one step of control.

        Args:
            state: Current state dictionary containing sensor data and robot state
            
        Returns:
            Tuple containing action, done flag, and success flag
        """
        self.state = state
        if self.initial_position is None:
            self.initial_position = self.state['object_position']
        if self.initial_size is None:
            self.initial_size = self.state['object_size']
        if self.mode == "collect":
            return self._step_collect(state)
        elif self.mode == "replay":
            return self._step_replay(state)
        else:
            return self._step_infer(state)

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
                action, record_array = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=self._pick_frame.quat(state['object_position']),
                    after_offset_z=0.3,
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
                    target_end_effector_orientation=self._pour_frame.quat(state['target_position'])
                )
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=record_array,
                    language_instruction=self.get_language_instruction()
                )
            
            return action, False, False

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

        self._last_failure_reason = f"PickPour {self.current_phase.value} failed" + (f": {self.last_error_info}" if self.last_error_info else "")
        print(f"{self.current_phase.value} task failed!")
        if self.last_error_info is not None:
            print(f"Phase failure details: {self.last_error_info}")
        self.data_collector.clear_cache()
        self._last_success = False
        self.current_phase = Phase.FINISHED
        return None, True, False

    def _step_replay(self, state):
        """Replay plays back the full recorded pick+pour trajectory.

        The base ``_step_replay`` never advances ``current_phase``, so without
        this override ``_check_phase_success`` would stay in the PICKING branch
        forever and replay would only ever validate the 0.12 m lift — never the
        strict pour gate. Advance PICKING->POURING once the source is lifted (the
        same threshold collect/infer use) so replay actually re-checks the pour in
        real physics. Pure recorded-action playback — no object binding.
        """
        if (self.current_phase == Phase.PICKING
                and self.initial_position is not None
                and state['object_position'][2] > self.initial_position[2] + 0.12):
            self.current_phase = Phase.POURING
        return super()._step_replay(state)

    def _step_infer(self, state):
        """Execute inference mode step."""
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        language_instruction = self.get_language_instruction()
        if language_instruction is not None:
            state['language_instruction'] = language_instruction
        else:
            state['language_instruction'] = "Pick up the graduated cylinder from the table and pour it into the big beaker"

        action = self.inference_engine.step_inference(state)
        success = self._check_phase_success()
        if success and self.current_phase == Phase.PICKING:
            print("Pick task success! Switching to pour...")
            self.current_phase = Phase.POURING
            self.inference_engine.trajectory_controller.reset()
        elif success and self.current_phase == Phase.POURING:
            print("Pour task success!")
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return None, True, True
               
        return action, False, False

    def get_language_instruction(self) -> Optional[str]:
        object_name = self.clean_object_name(self.state['object_name'])
        if 'beaker' in object_name:
            object_name = 'small beaker'
        if self.current_phase == Phase.PICKING or self._pour_prompt_mode == "pick":
            # infer.pour_phase_prompt == "pick" keeps the pick sentence through the
            # POURING phase: the exported training episodes label every frame
            # (pour segment included) with the pick instruction, so the pour
            # sentence below is out-of-vocabulary at eval time.
            return self._get_cached_instruction(
                'pickpour:picking',
                self._build_instruction_templates(
                    f"Pick up the {object_name}",
                    f"Pick up the {object_name} from the table and prepare it for pouring into the big beaker",
                ),
            )
        return self._get_cached_instruction(
            'pickpour:pouring',
            self._build_instruction_templates(
                f"Pour the contents of the {object_name} into the big beaker",
                f"Move the {object_name} over the big beaker and pour its contents carefully",
            ),
        )
