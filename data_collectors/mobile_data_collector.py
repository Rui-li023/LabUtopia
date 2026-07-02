from typing import Optional

import numpy as np

from .data_collector import DataCollector


class MobileDataCollector(DataCollector):
    """DataCollector for mobile-manipulator (Ridgebase) episodes.

    Records the unified 11-dim mobile layout instead of the Franka 8-dim one:

        indices 0-2:  base joints (x, y, theta)
        indices 3-9:  arm joints (panda_joint1..7)
        index 10:     gripper channel

    ``agent_pose[10]`` carries the physical gripper opening width (finger joint
    doubled, 0-0.08 m); ``actions[10]`` carries normalized closedness [0, 1].
    Every step also records an integer ``phase`` id (0 = navigate, 1 = pick,
    2 = carry-navigate, 3 = pour), written to the episode h5 as ``phase``.
    """

    GRIPPER_INDEX = 10  # finger joint in the 11-dim state layout
    WRIST_INDEX = 9     # panda_joint7 in the 11-dim layout

    def __init__(self, camera_configs, save_dir="output", max_episodes=10,
                 max_workers=4, compression=None):
        super().__init__(camera_configs, save_dir=save_dir,
                         max_episodes=max_episodes, max_workers=max_workers,
                         compression=compression)
        self.temp_phases: list[int] = []

    def cache_step(
        self,
        camera_images: dict,
        joint_angles: np.ndarray,
        action: Optional[np.ndarray] = None,
        language_instruction: Optional[str] = None,
        task_index: Optional[int] = None,
        phase: int = 0,
    ) -> None:
        """Cache one step in the 11-dim mobile layout with a phase label."""
        if task_index is None and language_instruction is not None:
            task_index = self.register_task_instruction(language_instruction)
        if self.task_instructions is None and language_instruction is not None:
            self.task_instructions = []

        for camera_name, image in camera_images.items():
            self.temp_cameras[camera_name].append(image)

        state = np.asarray(joint_angles, dtype=np.float32).copy()
        if len(state) > self.GRIPPER_INDEX:
            state[self.GRIPPER_INDEX] = float(state[self.GRIPPER_INDEX]) * 2.0
        self.temp_agent_pose.append(state)

        if action is not None:
            act = np.asarray(action, dtype=np.float32).copy()
            # Wrap the redundant wrist (panda_joint7, index 9 here) back to
            # within +-pi of the measured joint, mirroring the base collector's
            # index-6 handling for the 8-dim layout.
            if act.shape[0] > self.WRIST_INDEX and len(joint_angles) > self.WRIST_INDEX:
                ref = float(joint_angles[self.WRIST_INDEX])
                act[self.WRIST_INDEX] = ref + (float(act[self.WRIST_INDEX]) - ref + np.pi) % (2 * np.pi) - np.pi
            self.temp_actions.append(act)

        if language_instruction is not None:
            self.temp_language_instruction = language_instruction
            if isinstance(self.task_instructions, list) and language_instruction not in self.task_instructions:
                self.task_instructions.append(language_instruction)
        self.temp_task_indices.append(-1 if task_index is None else int(task_index))
        self.temp_phases.append(int(phase))

    def _extra_datasets(self) -> Optional[dict]:
        return {"phase": np.asarray(self.temp_phases, dtype=np.int32)}

    def write_cached_data(self, final_joint_positions=None) -> None:
        """Write the episode; every cached step must carry an explicit action.

        The base class's derived-action fallback assumes the 8-dim layout
        (it rescales index 7), which would corrupt an arm joint here, so it is
        forbidden for mobile episodes.
        """
        if len(self.temp_actions) != len(self.temp_agent_pose):
            raise ValueError(
                "MobileDataCollector requires an explicit action for every cached "
                f"step (got {len(self.temp_actions)} actions vs "
                f"{len(self.temp_agent_pose)} states)")
        super().write_cached_data(None)
        self.temp_phases = []

    def clear_cache(self) -> None:
        super().clear_cache()
        self.temp_phases = []
