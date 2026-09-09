"""Sub-stage keyframe capture for publication figure strips.

A task figure in the paper is a strip of frames — one per sub-stage of the
rollout ("approach", "grasp", "lift", "pour", ...).  Picking those frames by
timestamp is guesswork, because every episode randomizes object placement and
each atomic action runs for a different number of physics steps.

This module instead watches the *controller's own state machine* and writes a
frame every time its stage signature changes, so the strip is aligned with the
phase boundaries the controller actually executed.  Two granularities:

``phase``
    Only the task-level phase (``controller.current_phase``) plus the mobile
    navigation flags.  One frame per sub-stage of a composed task — the right
    choice for level-4 (10 sub-stages for liquid mixing) and level-5.

``event``
    Additionally keys on the ``_event`` counter of every atomic action
    controller held by the task controller.  One frame per state-machine
    transition — the right choice for a level-1/level-2 task whose whole
    rollout is one or two atomic actions.

Frames are written under ``<output_dir>/_attempt_<n>/`` as they are captured and
promoted to ``<output_dir>/frames/`` only when the episode succeeds, so a failed
attempt never contaminates the delivered strip.
"""

import os
import shutil
from typing import Any

import cv2
import numpy as np
from loguru import logger

# An atomic action controller is duck-typed rather than imported, to keep this
# module free of a controllers/ import (main.py builds the recorder before the
# controller factory runs).
_ATOMIC_MARKERS = ("_event", "_events_dt")

# Mobile manipulation splits navigate/dock/manipulate with plain booleans rather
# than a phase enum, so those flags are part of the signature too.
_FLAG_ATTRS = (
    "navigation_done",
    "_nav_reached",
    "_carry_nav_done",
    "_place_nav_done",
    "_corr_done",
)


def _is_atomic(obj: Any) -> bool:
    return all(hasattr(obj, marker) for marker in _ATOMIC_MARKERS)


def _phase_label(controller: Any) -> str:
    phase = getattr(controller, "current_phase", None)
    if phase is None:
        return ""
    return str(getattr(phase, "value", phase))


def stage_signature(controller: Any, granularity: str) -> tuple:
    """Return the controller's current sub-stage identity.

    Any change in the returned tuple marks a sub-stage boundary.
    """
    parts: list[str] = []
    phase = _phase_label(controller)
    if phase:
        parts.append(phase)
    for attr in _FLAG_ATTRS:
        value = getattr(controller, attr, None)
        if isinstance(value, bool):
            parts.append(f"{attr.lstrip('_')}={int(value)}")
    if granularity == "event":
        for name, obj in sorted(vars(controller).items()):
            if _is_atomic(obj):
                parts.append(f"{name}#{int(getattr(obj, '_event', -1))}")
    return tuple(parts)


def signature_label(signature: tuple) -> str:
    """Compact, filesystem-safe label for a signature tuple."""
    if not signature:
        return "stage"
    text = "-".join(str(part) for part in signature)
    text = text.replace("_controller", "").replace("controller", "")
    safe = "".join(ch if (ch.isalnum() or ch in "-_#=") else "_" for ch in text)
    return safe[:80] or "stage"


def frame_to_bgr(state: dict, camera_index: int) -> np.ndarray | None:
    """Extract one camera's RGB frame from a task state dict as BGR uint8."""
    frames = list((state.get("camera_display") or {}).values())
    if camera_index < 0 or camera_index >= len(frames):
        return None
    image = np.asarray(frames[camera_index])
    if image.ndim == 3 and image.shape[0] in (3, 4):
        image = image.transpose(1, 2, 0)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    image = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


class KeyframeRecorder:
    """Capture one clean PNG per controller sub-stage of a successful episode."""

    def __init__(
        self,
        output_dir: str,
        camera_index: int = 0,
        granularity: str = "event",
        min_steps: int = 0,
        keep_attempts: bool = False,
        warmup_steps: int = 30,
        every_steps: int = 0,
    ) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.camera_index = int(camera_index)
        self.granularity = granularity
        self.min_steps = int(min_steps)
        self.keep_attempts = bool(keep_attempts)
        self.warmup_steps = max(1, int(warmup_steps))
        self.every_steps = max(0, int(every_steps))
        self.completed = False
        self._attempt = 0
        self._attempt_dir: str | None = None
        self._rows: list[tuple[int, str, int, str]] = []
        self._last_signature: tuple | None = None
        self._last_saved_step = -(10**9)
        self._step = 0
        os.makedirs(self.output_dir, exist_ok=True)

    # ── Episode lifecycle ────────────────────────────────────────────────

    def begin_episode(self) -> None:
        """Start a fresh attempt; discards anything buffered from the last one."""
        if self.completed:
            return
        self._attempt += 1
        self._attempt_dir = os.path.join(self.output_dir, f"_attempt_{self._attempt:02d}")
        shutil.rmtree(self._attempt_dir, ignore_errors=True)
        os.makedirs(self._attempt_dir, exist_ok=True)
        self._rows = []
        self._last_signature = None
        self._last_saved_step = -(10**9)
        self._step = 0
        self._started = False

    def observe(self, controller: Any, state: dict) -> None:
        """Write a frame if the controller just crossed a sub-stage boundary.

        The frame captured at a transition is the *last* frame of the sub-stage
        that just finished — i.e. the pose the sub-action achieved, which is
        what a figure strip wants to show.
        """
        if self.completed or self._attempt_dir is None:
            return
        self._step += 1
        signature = stage_signature(controller, self.granularity)
        if self._step < self.warmup_steps:
            # RTX needs a few frames to converge after a reset; a frame read at
            # step 1 is half-denoised and unusable in a figure. Keep tracking the
            # signature so the first real transition is still caught.
            self._last_signature = signature
            return
        if not self._started:
            self._started = True
            self._last_signature = signature
            self._write(state, signature)
            return
        changed = signature != self._last_signature
        # Replay mode never advances a scripted controller's phases (it just
        # re-issues recorded joint targets), so a replayed rollout has no stage
        # boundaries to key on. `every_steps` gives it a uniform strip instead.
        due = self.every_steps > 0 and (self._step - self._last_saved_step) >= self.every_steps
        if not changed and not due:
            return
        if changed and not due and (self._step - self._last_saved_step) < self.min_steps:
            # Still record the signature so the *next* distinct stage is caught.
            self._last_signature = signature
            return
        self._last_signature = signature
        self._write(state, signature)

    def end_episode(self, state: dict | None, success: bool) -> str | None:
        """Write the final frame and promote the attempt if the episode passed."""
        if self.completed or self._attempt_dir is None:
            return None
        if success and state is not None:
            self._write(state, ("final",))
        if not success:
            logger.info(f"[keyframes] attempt {self._attempt} failed — discarding {len(self._rows)} frame(s)")
            if not self.keep_attempts:
                shutil.rmtree(self._attempt_dir, ignore_errors=True)
            self._attempt_dir = None
            return None

        frames_dir = os.path.join(self.output_dir, "frames")
        shutil.rmtree(frames_dir, ignore_errors=True)
        shutil.move(self._attempt_dir, frames_dir)
        manifest = os.path.join(self.output_dir, "frames.tsv")
        with open(manifest, "w", encoding="utf-8") as handle:
            handle.write("index\tfile\tstep\tsignature\n")
            for index, filename, step, signature in self._rows:
                handle.write(f"{index}\t{filename}\t{step}\t{signature}\n")
        self.completed = True
        self._attempt_dir = None
        logger.success(f"[keyframes] {len(self._rows)} sub-stage frame(s) -> {frames_dir}")
        return frames_dir

    # ── Internals ────────────────────────────────────────────────────────

    def _write(self, state: dict, signature: tuple) -> None:
        image = frame_to_bgr(state, self.camera_index)
        if image is None:
            logger.warning(f"[keyframes] camera index {self.camera_index} produced no frame; skipping this sub-stage")
            return
        index = len(self._rows)
        label = signature_label(signature)
        filename = f"{index:02d}_{label}.png"
        path = os.path.join(self._attempt_dir, filename)
        if not cv2.imwrite(path, image):
            logger.error(f"[keyframes] failed to write {path}")
            return
        self._rows.append((index, filename, self._step, " | ".join(str(p) for p in signature)))
        self._last_saved_step = self._step
        logger.info(f"[keyframes] {filename} ({image.shape[1]}x{image.shape[0]}) step={self._step} stage={signature}")
