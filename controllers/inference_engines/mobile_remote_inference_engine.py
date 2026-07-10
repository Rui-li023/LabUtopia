from typing import Any, Dict, List, Optional

import numpy as np

from .remote_inference_engine import RemoteInferenceEngine


class MobileRemoteInferenceEngine(RemoteInferenceEngine):
    """Remote inference for Level-5 mobile manipulation (Ridgebase, 11-dim).

    Differs from the static-arm ``RemoteInferenceEngine`` in two ways that the
    shared pipeline gets wrong for a full-body mobile policy:

    1. **State layout.** The policy is trained on the unified 11-dim state
       ``[base x, y, theta] + 7 arm joints + finger1 opening`` (``_state11``).
       The base engine's ``update_observations`` instead drops the last channel
       and doubles index 7 (a single-arm gripper-width fix) — wrong here. We
       send the raw 11-dim state the dataset recorded.

    2. **No Franka trajectory controller.** The predicted action is an 11-dim
       vector whose base dims 0:3 are BODY-frame deltas; it must be integrated
       onto the measured base pose and applied to all 12 DOFs by the mobile
       controller (``_apply_action11``), NOT interpolated through the arm
       trajectory controller. ``step_inference`` therefore bypasses the
       trajectory controller and returns the raw 11-dim action, one per frame,
       from an internally buffered action chunk.
    """

    def _init_inference_engine(self) -> None:
        super()._init_inference_engine()
        self._action_queue: List[np.ndarray] = []

    def update_observations(self, state: Dict[str, Any]) -> None:
        """Append the current cameras + full 11-dim state to the histories."""
        for cam_name, image in state["camera_data"].items():
            if cam_name in self.camera_to_obs:
                self.obs_history_dict[self.camera_to_obs[cam_name]].append(image)

        # Full 11-dim unified state, exactly as recorded at collect time
        # (base 3 + arm 7 + gripper). MobileDataCollector stores the gripper
        # channel (index 10) as finger1 x 2 (total opening width, ~0.044-0.08),
        # so the policy trained on that scale — double it here to match, else
        # the policy sees half the gripper state it was trained on and the
        # grasp/close behavior is corrupted.
        pose = np.asarray(state["agent_pose"], dtype=np.float32).copy()
        if pose.shape[0] > 10:
            pose[10] *= 2.0
        self.obs_history_pose.append(pose)

        self.language_instruction = state.get("language_instruction", "") or ""

    def step_inference(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Return one raw 11-dim action per frame from a buffered chunk.

        Refills the buffer with a fresh action chunk from the policy whenever it
        drains and the observation history is complete. Returns ``None`` while
        the history is still filling (n_obs_steps warm-up).
        """
        self.update_observations(state)

        if not self._action_queue and self._check_histories_complete():
            obs_dict = self._prepare_observation_dict()
            actions = np.asarray(self._predict_action(obs_dict, self.language_instruction))
            if actions.ndim == 1:
                actions = actions[None, :]
            chunk_len = int(getattr(self.cfg.infer, "action_chunk_len", 10))
            self._action_queue = [np.asarray(a, dtype=np.float32) for a in actions[:chunk_len]]

        if self._action_queue:
            return self._action_queue.pop(0)
        return None

    def reset(self) -> None:
        super().reset()
        self._action_queue = []
