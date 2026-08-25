import random
import numpy as np
from typing import Any, Dict, Optional
from loguru import logger

from .base_task import BaseTask


class PressTask(BaseTask):
    """Button-press task with two distractor buttons.

    The three buttons are joint-constrained inside ``/World/instrument``;
    each has a prismatic joint that locks lateral position. To randomize
    rest position per episode we shift each joint's anchor on the
    instrument-side body along Y (before ``world.reset`` so PhysX rebakes
    the constraint).
    """

    _INSTRUMENT_POSITION = np.array([0.73, -0.1, 0.64])
    # Three non-overlapping Y buckets (~5 cm wide, ~10 cm centre-to-centre)
    # so the three buttons never collide regardless of shuffle order.
    _Y_BUCKETS = [
        (0.075, 0.125),   # +Y end
        (-0.025, 0.025),  # middle
        (-0.125, -0.075), # -Y end
    ]

    def __init__(self, cfg: Any, world: Any, stage: Any, robot: Any) -> None:
        super().__init__(cfg, world, stage, robot)
        # Optional per-episode instrument placement (``task.instrument_position_range``
        # with x/y/z [lo, hi] lists). Without it the instrument sits at the single
        # hard-coded spot for all 50 episodes, so every demo shows the buttons at the
        # same place and only their y ordering ever changes — the position half of the
        # task is not generalized at all.
        task_cfg = getattr(cfg, "task", None)
        self._instrument_range = getattr(task_cfg, "instrument_position_range", None) if task_cfg else None
        self.object_utils.set_object_position(
            object_path=self.cfg.instrument_path,
            position=self._INSTRUMENT_POSITION,
        )
        self.target_button_path      = self.cfg.target_button_path
        self.distractor_button1_path = self.cfg.distractor_button1_path
        self.distractor_button2_path = self.cfg.distractor_button2_path

        self.button_paths = [
            self.target_button_path,
            self.distractor_button1_path,
            self.distractor_button2_path,
        ]
        self.joint_paths = list(self.cfg.button_joint_paths)
        assert len(self.joint_paths) == 3, "button_joint_paths must have 3 entries"
        # If the joint's body is under a scaled xform (e.g. instrument with
        # scale=0.001), world-space offsets must be divided by that scale
        # before being added to localPos.
        self._joint_lp_scale = float(getattr(self.cfg, "button_joint_localpos_scale", 1.0))

        # Pick the side (0 or 1) whose body is the instrument (static anchor),
        # not the moving button. We modify that side's localPos so the button
        # rest world position shifts.
        self._joint_anchor_side = {}
        self._joint_base_local_pos = {}
        for jp in self.joint_paths:
            b0, b1 = self.object_utils.get_joint_bodies(jp)
            # Heuristic: if body1 path contains 'button', body0 is instrument.
            if b1 and "button" in b1.lower():
                side = 0
            elif b0 and "button" in b0.lower():
                side = 1
            else:
                side = 0
            self._joint_anchor_side[jp] = side
            self._joint_base_local_pos[jp] = self.object_utils.get_joint_local_pos(jp, side=side)
            logger.info(
                f"[press task] joint={jp} body0={b0} body1={b1} -> "
                f"will modify localPos{side}, base={self._joint_base_local_pos[jp]}"
            )

    def _sample_instrument_position(self) -> np.ndarray:
        """This episode's instrument placement, or the fixed spot if unconfigured."""
        if self._instrument_range is None:
            return self._INSTRUMENT_POSITION.copy()
        base = self._INSTRUMENT_POSITION
        return np.array(
            [
                random.uniform(*getattr(self._instrument_range, axis, (base[i], base[i])))
                for i, axis in enumerate(("x", "y", "z"))
            ],
            dtype=np.float32,
        )

    def reset(self) -> None:
        # Move the instrument BEFORE world.reset() for the same reason as the
        # joint anchors below: the buttons' prismatic joints are anchored to it,
        # and PhysX only rebakes joint frames on simulation re-init.
        instrument_position = self._sample_instrument_position()
        self.object_utils.set_object_position(
            object_path=self.cfg.instrument_path,
            position=instrument_position,
        )

        # Mutate joint anchors BEFORE world.reset() — PhysX rebakes joint
        # frames on simulation re-init, so the new USD values need to be in
        # place first.
        y_offsets = [random.uniform(*b) for b in self._Y_BUCKETS]
        random.shuffle(y_offsets)

        anchors: dict[str, list[float]] = {}
        for joint_path, dy in zip(self.joint_paths, y_offsets):
            base = self._joint_base_local_pos.get(joint_path)
            if base is None:
                continue
            side = self._joint_anchor_side[joint_path]
            new_lp = base.copy()
            local_dy = dy / self._joint_lp_scale if self._joint_lp_scale != 0 else dy
            new_lp[1] = base[1] + local_dy
            self.object_utils.set_joint_local_pos(joint_path, new_lp, side=side)
            anchors[joint_path] = [float(v) for v in new_lp]
            logger.info(
                f"[press task] {joint_path} localPos{side}: {base.tolist()} -> {new_lp.tolist()} "
                f"(world_dy={dy:.3f}, local_dy={local_dy:.3f})"
            )

        super().reset()
        self.robot.initialize()

        # super().reset() re-initializes _episode_init_state, so record the
        # anchors only now. They ride in init_state extra (JSON) and are
        # restored by reset_with_init_state before world.reset() rebakes joints.
        self._episode_init_state["extra"]["button_joint_anchors"] = anchors
        self._episode_init_state["extra"]["instrument_position"] = [
            float(v) for v in instrument_position
        ]

        for path in self.button_paths:
            self._record_object_pose(path)

    def reset_with_init_state(self, init_state: dict) -> None:
        # Restore the recorded per-episode button joint anchors BEFORE
        # super().reset_with_init_state() — that calls world.reset(), which is
        # when PhysX rebakes joint frames from the USD values (same ordering
        # collect uses in reset()). An earlier attempt restored anchors with the
        # wrong timing and corrupted the buttons; the ordering is the fix.
        instrument_position = init_state.get("extra", {}).get("instrument_position")
        if instrument_position is not None:
            self.object_utils.set_object_position(
                object_path=self.cfg.instrument_path,
                position=np.asarray(instrument_position, dtype=np.float32),
            )
        anchors = init_state.get("extra", {}).get("button_joint_anchors") or {}
        for joint_path, lp in anchors.items():
            side = self._joint_anchor_side.get(joint_path)
            if side is None:
                continue
            self.object_utils.set_joint_local_pos(
                joint_path, np.asarray(lp, dtype=np.float32), side=side
            )
            logger.info(f"[press task] replay restored anchor {joint_path} localPos{side}={lp}")
        super().reset_with_init_state(init_state)

    def step(self) -> Optional[Dict[str, Any]]:
        self.frame_idx += 1
        if not self.check_frame_limits(max_steps=1000):
            return None

        # Use the rigid button mesh, not the parent xform, as the target —
        # the parent never moves and sits at a different X than the actual
        # contact face, which makes the press controller aim at the wrong
        # point.
        rigid_button_path = self.target_button_path + "/button"
        return self.get_basic_state_info(
            object_path=self.target_button_path,
            additional_info={
                "object_position": self.object_utils.get_object_xform_position(rigid_button_path),
            },
        )
