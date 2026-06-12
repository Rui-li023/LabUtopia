import random
from enum import Enum
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .base_controller import BaseController


class Phase(Enum):
    PICKING = "picking"
    PLACING = "placing"
    FINISHED = "finished"


class FlaskToCorkTaskController(BaseController):
    """Controller for the Level-2 flask-to-cork-ring pick-and-place task.

    Phase 1 – PICKING: grasp the round-bottom flask at its neck.
    Phase 2 – PLACING: lower the flask into the cork ring.
    """

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
        self.initial_size = None
        self.current_phase = Phase.PICKING
        self.last_error_info = None
        self._phase_instructions: dict[Phase, str] = {}
        self._phase_task_indices: dict[Phase, int] = {}

        # Per-finger grasp target for the round-bottom flask neck (Franka:
        # 0 = closed, 0.04 = open). Too small → the fingers slam past contact and
        # clip through the neck mesh (physics ejects the flask); too large → no
        # contact and it drops. A sweep over 0.004–0.012 put picking success at
        # 20% → 47% → 87% (0.008) → 87% (0.010) → 93% (0.012); 0.008–0.012 hold
        # the flask with zero picking failures, so 0.010 is the robust default.
        # Overridable via cfg.task.gripper_distance for further tuning.
        task_cfg = getattr(cfg, "task", None)
        self._gripper_distance = float(getattr(task_cfg, "gripper_distance", 0.010))

    # ------------------------------------------------------------------ setup

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02],
        )
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            robot=robot,
        )
        self.active_controller = self.pick_controller

    # ------------------------------------------------------------------ reset

    def reset(self):
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self.last_error_info = None
        self._phase_instructions = {}
        self._phase_task_indices = {}

        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pick_controller.reset()
            self.place_controller.reset()
        elif self.mode == "infer":
            self.inference_engine.reset()

    # ---------------------------------------------------------- success checks

    def _check_success(self) -> bool:
        # In replay the phase machine (which only advances in _step_collect) is
        # frozen at PICKING, so a phase-based check would score the recorded
        # pick+place trajectory against "is the flask lifted" — which is false
        # once the flask has been set down at the end of the episode. Score
        # replay against the final task goal instead: flask resting on target,
        # using the same thresholds as the PLACING phase check.
        if self.mode == "replay":
            return self._check_placed_on_target()
        return self._check_phase_success()

    def _check_placed_on_target(self) -> bool:
        object_pos = self.state["object_position"]
        target_pos = self.state["target_position"]
        xy_dist = float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
        z_drop = float(abs(object_pos[2] - self.initial_position[2]))
        # Replay leaves no failure reason (base _step_replay), so record the
        # placement metrics for diagnosis.
        self._last_failure_reason = (
            f"replay placement: xy_dist={xy_dist:.3f} (<0.08) "
            f"z_drop={z_drop:.3f} (<0.15)"
        )
        return xy_dist < 0.08 and z_drop < 0.15

    def _check_phase_success(self) -> bool:
        object_pos = self.state["object_position"]
        target_pos = self.state["target_position"]

        if self.current_phase == Phase.PICKING:
            # 5 cm lift is enough to confirm the flask cleared its support.
            required_height = self.initial_position[2] + 0.05
            success = object_pos[2] > required_height
            if not success:
                self.last_error_info = {
                    'phase': 'PICKING',
                    'current_height': float(object_pos[2]),
                    'required_height': float(required_height),
                    'height_diff': float(object_pos[2] - required_height),
                }
            return success

        if self.current_phase == Phase.PLACING:
            xy_dist = float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
            z_drop = float(abs(object_pos[2] - self.initial_position[2]))
            success = xy_dist < 0.08 and z_drop < 0.15
            if not success:
                self.last_error_info = {
                    'phase': 'PLACING',
                    'xy_distance': xy_dist,
                    'xy_threshold': 0.08,
                    'z_drop_from_initial': z_drop,
                    'z_threshold': 0.15,
                }
            return success

        return False

    # ---------------------------------------------------------- language utils

    def _object_display_name(self, path: str) -> str:
        return path.split("/")[-1].replace("_", " ")

    def _sample_phase_instruction(self, phase: Phase) -> str:
        source = self._object_display_name(self.cfg.task.obj_paths[0]["path"])
        target = self._object_display_name(self.cfg.task.obj_paths[1]["path"])
        support = self._object_display_name(self.cfg.task.obj_paths[2]["path"])
        if phase == Phase.PICKING:
            templates = self._build_instruction_templates(
                f"Pick up the {source} from the {support}",
                f"Pick up the {source} from the {support} and lift it clear of the surface",
            )
        else:
            templates = self._build_instruction_templates(
                f"Place the {source} on the {target}",
                f"Move the {source} to the {target} and set it down carefully",
            )
        return random.choice(templates)

    def get_language_instruction(self) -> str:
        if self.current_phase not in self._phase_instructions:
            self._phase_instructions[self.current_phase] = self._sample_phase_instruction(
                self.current_phase
            )
        self._language_instruction = self._phase_instructions[self.current_phase]
        return self._language_instruction

    def get_task_index(self) -> Optional[int]:
        if self.mode != "collect" or self.current_phase == Phase.FINISHED:
            return None
        if self.current_phase not in self._phase_task_indices:
            instruction = self.get_language_instruction()
            self._phase_task_indices[self.current_phase] = (
                self.data_collector.register_task_instruction(instruction)
            )
        return self._phase_task_indices[self.current_phase]

    # -------------------------------------------------------------------- step

    def step(self, state):
        self.state = state
        if self.initial_position is None:
            self.initial_position = np.array(state["object_position"])
        if self.initial_size is None:
            self.initial_size = state["object_size"]

        if self.mode == "collect":
            return self._step_collect(state)
        if self.mode == "replay":
            # Replay is handled by BaseController (trajectory playback + success
            # check); this override only exists to seed initial_position/size.
            return self._step_replay(state)
        return self._step_infer(state)

    # ------------------------------------------------------- collect-mode step

    def _step_collect(self, state):
        success = self._check_phase_success()

        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action, record_array = self._forward_active(state)

            if "camera_data" in state:
                self.data_collector.cache_step(
                    camera_images=state["camera_data"],
                    joint_angles=state["joint_positions"][:-1],
                    action=record_array,
                    language_instruction=self.get_language_instruction(),
                    task_index=self.get_task_index(),
                )
            return action, False, False

        # active controller finished – evaluate success
        if success:
            if self.current_phase == Phase.PICKING:
                print("Flask picked! Switching to place phase.")
                self.current_phase = Phase.PLACING
                self.active_controller = self.place_controller
                return None, False, False

            if self.current_phase == Phase.PLACING:
                print("Flask placed on cork ring – task success.")
                self._last_failure_reason = ""
                self.data_collector.write_cached_data(state["joint_positions"][:-1])
                self._last_success = True
                self.current_phase = Phase.FINISHED
                return None, True, True

        # controller done but success not achieved
        detail = f" details: {self.last_error_info}" if self.last_error_info else ""
        self._last_failure_reason = (
            f"FlaskToCork {self.current_phase.value} phase failed: "
            f"success check did not pass after controller finished{detail}"
        )
        print(f"{self.current_phase.value} phase failed.")
        if self.last_error_info:
            print(f"Phase failure details: {self.last_error_info}")
        self.data_collector.clear_cache()
        self._last_success = False
        self.current_phase = Phase.FINISHED
        return None, True, False

    def _forward_active(self, state):
        """Dispatch forward() call to the currently active atomic controller."""
        # Approach orientation: wrist vertical, gripper aligned for neck grasp
        ee_orient = R.from_euler("xyz", np.radians([0, 90, 30])).as_quat()

        if self.current_phase == Phase.PICKING:
            return self.pick_controller.forward(
                picking_position=state["object_position"],
                current_joint_positions=state["joint_positions"],
                object_size=state["object_size"],
                object_name=state["object_name"],
                gripper_control=self.gripper_control,
                gripper_position=state["gripper_position"],
                end_effector_orientation=ee_orient,
                pre_offset_x=0.04,
                pre_offset_z=0.02,
                after_offset_z=0.25,
                # Round-bottom flask: close to the neck width (per-finger target)
                # instead of a full 0/1 close, which ejected the flask. Tuned via
                # cfg.task.gripper_distance (see __init__); default 0.010.
                gripper_distances=self._gripper_distance,
            )

        # PLACING
        return self.place_controller.forward(
            place_position=state["target_position"],
            current_joint_positions=state["joint_positions"],
            gripper_control=self.gripper_control,
            end_effector_orientation=ee_orient,
            gripper_position=state["gripper_position"],
            place_offset_z=0.13,
        )

    # ------------------------------------------------------- infer-mode step

    def _step_infer(self, state):
        self.state = state
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        state["language_instruction"] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()

    def is_success(self) -> bool:
        object_pos = self.state["object_position"]
        target_pos = self.state["target_position"]
        if (
            np.linalg.norm(object_pos[:2] - target_pos[:2]) < 0.08
            and abs(object_pos[2] - self.initial_position[2]) < 0.15
            and np.linalg.norm(self.state["gripper_position"] - object_pos) > 0.05
        ):
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return True
        return False
