"""Read-only physical grasp observations, independent of controller phases."""

from functools import wraps
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation


def _vector(value: Any, size: int) -> list[float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite {size}-component telemetry vector")
    return array.tolist()


def capture_grasp_sample(controller: Any, state: dict) -> dict:
    """Copy physical poses before the next command; never retain mutable state."""
    fingers = np.asarray(controller.robot.gripper.joint_dof_indicies, dtype=int)
    objects = state.get("pipettes", [state])
    return {
        "type": "sample",
        "episode": int(controller._episode_num),
        "mode": controller.mode,
        "replay_action_step": int(getattr(controller, "_current_action_step", 0)),
        "closed_command": int(controller.robot._gripper_state) == 1,
        "finger_positions": _vector(np.asarray(state["joint_positions"])[fingers], 2),
        "tcp": _vector(state["gripper_position"], 3),
        # Task reports the actual USD panda_hand frame in xyzw order.
        "hand_quaternion_xyzw": _vector(state["physical_hand_quaternion"], 4),
        "objects": [
            {
                "path": obj["object_path"],
                "quaternion_xyzw": _vector(obj["object_quaternion"], 4),
                "grasp_position": _vector(obj["object_grasp_position"], 3),
            }
            for obj in objects
        ],
    }


def observed_step(original: Callable, emit: Callable) -> Callable:
    """Wrap only observation/serialization; delegate the exact original action."""

    @wraps(original)
    def step(controller: Any, state: dict):
        sample = capture_grasp_sample(controller, state)
        result = original(controller, state)
        sample.update(done=bool(result[1]), success=bool(result[2]))
        emit(sample)
        return result

    return step


def summarize_grasp_samples(samples: list[dict]) -> dict:
    """Measure fixed-object motion in the physical hand frame while closed.

    The contact reference requires 10 stable jaw samples. A second reference
    requires 10 consecutive stationary jaw/object/TCP samples, matching the
    velocity-mode close gate. This is an
    observational definition, not the controller's phase-5 reference and not
    a task success gate. Missing references remain missing, never zero slip.
    """
    segments = []
    current = None
    previous = None
    contact_reference = settled_reference = None
    jaw_stable = settled = 0
    reference_tcp = reference_hand = None
    reference_window_open = True

    def finish(released: bool) -> None:
        nonlocal current, previous, contact_reference, settled_reference, jaw_stable, settled
        nonlocal reference_tcp, reference_hand, reference_window_open
        if current is not None:
            current["release_observed"] = released
            segments.append(current)
        current = previous = contact_reference = settled_reference = None
        jaw_stable = settled = 0
        reference_tcp = reference_hand = None
        reference_window_open = True

    for sample in samples:
        if sample.get("type") != "sample":
            continue
        if current is not None and sample["episode"] != current["episode"]:
            finish(False)
        if not sample["closed_command"]:
            finish(True)
            continue
        tcp = np.asarray(sample["tcp"])
        if current is None:
            nearest = min(sample["objects"], key=lambda obj: np.linalg.norm(np.asarray(obj["grasp_position"]) - tcp))
            current = {
                "episode": sample["episode"],
                "object_path": nearest["path"],
                "selection_distance_m": float(np.linalg.norm(np.asarray(nearest["grasp_position"]) - tcp)),
                "closed_samples": 0,
                "contact_reference_sample": None,
                "settled_reference_sample": None,
                "peak_contact_rotation_deg": None,
                "peak_settled_rotation_deg": None,
                "peak_grasp_distance_m": 0.0,
            }
        obj = next(obj for obj in sample["objects"] if obj["path"] == current["object_path"])
        position = np.asarray(obj["grasp_position"])
        rotation = Rotation.from_quat(obj["quaternion_xyzw"])
        hand = Rotation.from_quat(sample["hand_quaternion_xyzw"])
        relative = hand.inv() * rotation
        gap = sum(sample["finger_positions"])
        current["closed_samples"] += 1
        current["peak_grasp_distance_m"] = max(current["peak_grasp_distance_m"], float(np.linalg.norm(position - tcp)))
        stable_gap = previous is not None and 0.016 <= gap <= 0.036 and abs(gap - previous[0]) < 0.0001
        jaw_stable = jaw_stable + 1 if stable_gap else 0
        stationary = stable_gap and (
            np.linalg.norm(position - previous[1]) < 0.0001
            and np.linalg.norm(tcp - previous[2]) < 0.0001
            and (rotation * previous[3].inv()).magnitude() < np.radians(0.05)
        )
        settled = settled + 1 if stationary else 0
        if contact_reference is None and jaw_stable >= 10:
            contact_reference = relative
            current["contact_reference_sample"] = current["closed_samples"]
            reference_tcp, reference_hand = tcp.copy(), hand
        if reference_tcp is not None and (
            np.linalg.norm(tcp - reference_tcp) > 0.001 or (hand * reference_hand.inv()).magnitude() > np.radians(1)
        ):
            # Never establish a new reference during carrying or insertion.
            reference_window_open = False
        if settled_reference is None and settled >= 10 and reference_window_open:
            settled_reference = relative
            current["settled_reference_sample"] = current["closed_samples"]
        for key, reference in (
            ("peak_contact_rotation_deg", contact_reference),
            ("peak_settled_rotation_deg", settled_reference),
        ):
            if reference is not None:
                angle = float(np.degrees((relative * reference.inv()).magnitude()))
                current[key] = max(current[key] or 0.0, angle)
        previous = (gap, position, tcp, rotation)
    finish(False)
    values = [s["peak_settled_rotation_deg"] for s in segments if s["peak_settled_rotation_deg"] is not None]
    contact_values = [s["peak_contact_rotation_deg"] for s in segments if s["peak_contact_rotation_deg"] is not None]
    return {
        "definition": "physical hand-relative rotation; references at 10 jaw-stable / 10 pose-stable samples before transport",
        "controller_phase_reference": False,
        "closed_segments": len(segments),
        "segments_with_settled_reference": len(values),
        "peak_settled_rotation_deg": max(values) if values else None,
        "peak_contact_rotation_deg": max(contact_values) if contact_values else None,
        "segments": segments,
        "note": "Diagnostic only. Missing/late references and unreleased segments must be inspected; no acceptance claim.",
    }
