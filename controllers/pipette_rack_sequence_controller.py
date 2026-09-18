"""One episode contains all configured pick/place cycles and one final commit."""

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R

from .flask_to_cork_controller import Phase
from .pipette_rack_controller import PipetteRackTaskController


class PipetteRackSequenceTaskController(PipetteRackTaskController):
    def __init__(self, cfg, robot):
        if len(cfg.task.pipette_paths) not in (3, 4):
            raise ValueError("Level 4 requires three or four configured pipettes")
        self._active_pipette = 0
        self._scene_state = None
        self._all_previous_positions = None
        self._all_initial_sizes = None
        super().__init__(cfg, robot)

    @property
    def pipette_count(self) -> int:
        return len(self.cfg.task.pipette_paths)

    def reset(self):
        super().reset()
        self._active_pipette = 0
        self._scene_state = None
        self._all_previous_positions = None
        self._all_initial_sizes = None

    def step(self, state):
        pipettes = state.get("pipettes", [])
        if (
            len(pipettes) != self.pipette_count
            or len({p["object_path"] for p in pipettes}) != self.pipette_count
            or len({p["slot_index"] for p in pipettes}) != self.pipette_count
            or [p["object_path"] for p in pipettes] != list(self.cfg.task.pipette_paths)
            or [p["slot_index"] for p in pipettes] != list(self.cfg.task.slot_indices)
        ):
            raise ValueError("Level 4 observations must match all configured distinct pipettes and slots")
        self._scene_state = state
        if self._all_initial_sizes is None:
            self._all_initial_sizes = [np.asarray(p["object_size"]).copy() for p in pipettes]
        selected = {**state, **pipettes[self._active_pipette]}
        return super().step(selected)

    def _check_success(self) -> bool:
        return self._check_all_placed()

    def _check_placed_on_target(self) -> bool:
        if self.mode != "collect" or self._active_pipette == self.pipette_count - 1:
            return self._check_all_placed()
        return super()._check_placed_on_target()

    def _check_all_placed(self) -> bool:
        if self._scene_state is None or self._all_initial_sizes is None:
            return False
        pipettes = self._scene_state["pipettes"]
        if len(pipettes) != self.pipette_count or len(self._all_initial_sizes) != self.pipette_count:
            self._last_failure_reason = "Level 4 success requires every configured pipette"
            return False
        positions = np.asarray([p["object_position"] for p in pipettes])
        previous = self._all_previous_positions
        self._all_previous_positions = positions.copy()
        if previous is None:
            return False
        tcp = self._scene_state["gripper_position"]
        for index, (pipette, initial_size) in enumerate(zip(pipettes, self._all_initial_sizes)):
            position, goal = positions[index], pipette["target_seat_position"]
            axis = np.asarray(pipette["object_axis"])
            target_axis = np.asarray(self._scene_state["target_axis"])
            cosine = float(np.dot(axis, target_axis) / max(np.linalg.norm(axis) * np.linalg.norm(target_axis), 1e-9))
            valid = (
                self._placement_xy_ok(position, goal)
                and abs(position[2] - goal[2]) < float(self.cfg.task.placement_z_tolerance)
                and pipette["object_size"][2] > 0.8 * max(initial_size)
                and cosine > np.cos(np.radians(20))
                and np.linalg.norm(tcp - position) > 0.07
                and np.linalg.norm(tcp - pipette["object_grasp_position"]) > 0.07
                and np.linalg.norm(position - previous[index]) < 0.001
            )
            if not valid:
                self._last_failure_reason = f"Level 4 pipette {index + 1}/{self.pipette_count} not stably released in slot {pipette['slot_index']}"
                return False
        self._last_failure_reason = ""
        return True

    def _on_placement_complete(self, state):
        if self._active_pipette == self.pipette_count - 1:
            return super()._on_placement_complete(state)
        logger.info(
            "Level 4: pipette {}/{} placed; retaining the same scene and episode cache",
            self._active_pipette + 1,
            self.pipette_count,
        )
        self._cache_hold_step(state)
        held_targets = self.place_controller._last_record_positions
        held_targets = None if held_targets is None else held_targets.copy()
        self._active_pipette += 1
        # Only reset action state. BaseController.reset would flush the episode;
        # task.reset would teleport the already placed pipettes back to the table.
        self.pick_controller.reset()
        self.pick_controller._last_record_positions = held_targets
        self.place_controller.reset()
        self.rmp_controller.reset()
        self.place_controller._cspace_controller = self.rmp_controller
        self.pick_controller._cspace_controller = self.rmp_controller
        self._lift_motion_configured = False
        self._pick_target_orientation = None
        self._close_previous_pose = None
        self._planned_pick_orientation = None
        self._preflight_wait_steps = 0
        self._transit_phase = self._transit_steps = self._transit_stable = 0
        self.active_controller = self.pick_controller
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self._grasp_offset = None
        self._previous_object_position = None
        self._settle_steps = self._placement_steps = 0
        self._placement_rotation = R.identity()
        self._placement_offset_refined = self._alignment_done = False
        self._diagnostic_steps = self._alignment_steps = self._grasp_trace_steps = 0
        self._grasp_reference = None
        self._phase_instructions = {}
        self._phase_task_indices = {}
        self.last_error_info = None
        self._last_failure_reason = ""
        return None, False, False

    def _sample_phase_instruction(self, phase):
        action = "Pick up" if phase == Phase.PICKING else "Place"
        count = "three" if self.pipette_count == 3 else "four"
        return f"{action} pipette {self._active_pipette + 1} of {count} and put all {count} pipettes into separate rack slots"

    def get_language_instruction(self):
        if self.mode == "infer":
            count = "three" if self.pipette_count == 3 else "four"
            return f"Pick up all {count} pipettes and place them upright in {count} separate rack slots"
        return super().get_language_instruction()
