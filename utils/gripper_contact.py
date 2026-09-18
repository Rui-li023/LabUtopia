"""Validated finite contact settings for the pipette's standard Franka Hand."""

import math
from dataclasses import asdict, dataclass
from typing import Any

from utils.franka_mimic_force import validate_franka_drive_force


@dataclass(frozen=True)
class CompliantFingerPadParameters:
    """Finite rubber-pad approximation, not a measured material calibration."""

    min_torsional_radius: float = 0.004
    stiffness: float = 60000.0
    damping: float = 100.0
    contact_offset: float = 0.0005
    static_friction: float = 0.9
    dynamic_friction: float = 0.7

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Finger-pad parameters must be finite")
        if not 0 <= self.min_torsional_radius <= 0.004:
            raise ValueError("Finger-pad contact radius must be within 4 mm")
        if not 35000 <= self.stiffness <= 1e7 or not 0 <= self.damping <= 1000:
            raise ValueError("Finger-pad spring/damping is outside the finite-contact envelope")
        if not 0.0001 <= self.contact_offset <= 0.005:
            raise ValueError("Finger-pad contact offset must be in [0.1, 5] mm")
        if not 0 <= self.dynamic_friction <= self.static_friction <= 1.2:
            raise ValueError("Finger-pad friction must satisfy 0 <= dynamic <= static <= 1.2")


def finger_pad_snapshot(cfg: Any) -> dict[str, float]:
    options = getattr(getattr(cfg, "task", None), "gripper_contact", None)
    if options is None:
        return {}
    parameters = CompliantFingerPadParameters(**dict(options))
    force = float(getattr(cfg.task, "gripper_drive_force", 30.0))
    if force / parameters.stiffness > 0.001:
        raise ValueError("Finger-pad static full-force compression must not exceed 1 mm")
    return asdict(parameters)


@dataclass(frozen=True)
class PipetteContactParameters:
    closing_speed: float = 0.05
    closing_force_per_finger: float = 30.0
    static_friction: float = 0.9
    dynamic_friction: float = 0.7
    contact_stiffness: float = 0.0
    contact_damping: float = 0.0
    contact_offset: float = 0.0  # zero preserves the robot asset's offsets
    opening_speed: float | None = None  # None retains the closing-speed limit
    force_semantics: str = "legacy"  # explicit single_mimic_drive allows a calibrated actuator cap

    @property
    def closing_force_drive_cap(self) -> float:
        """The legacy field name does not describe measured force at either jaw."""
        return self.closing_force_per_finger

    def __post_init__(self) -> None:
        values = (
            self.closing_speed,
            self.closing_force_per_finger,
            self.static_friction,
            self.dynamic_friction,
            self.contact_stiffness,
            self.contact_damping,
            self.contact_offset,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Pipette contact parameters must be finite")
        if not 0 < self.closing_speed <= 0.05:
            raise ValueError("Franka Hand closing speed must be in (0, 0.05] m/s")
        if self.opening_speed is not None and (
            not math.isfinite(self.opening_speed) or not 0 < self.opening_speed <= 0.05
        ):
            raise ValueError("Franka Hand opening speed must be in (0, 0.05] m/s")
        # Preserve the conservative legacy bound. Larger caps need explicit
        # single-actuator semantics AND the task's read-only topology guard.
        validate_franka_drive_force(self.closing_force_drive_cap, self.force_semantics)
        # Dry contact engineering-estimate envelope, not material calibration.
        if not 0 <= self.dynamic_friction <= self.static_friction <= 1.2:
            raise ValueError("Pad friction must satisfy 0 <= dynamic <= static <= 1.2")
        # A single contact carrying the full 35 N finger cap should compress
        # by <= 1 mm in the static force-spring estimate. This is not a bound
        # on transient impacts; contact penetration still needs simulation.
        minimum_stiffness = max(35000.0, self.closing_force_drive_cap / 0.001)
        if self.contact_stiffness != 0 and not minimum_stiffness <= self.contact_stiffness <= 1e7:
            raise ValueError(f"Pad contact stiffness must be zero (rigid) or in [{minimum_stiffness:g}, 1e7] N/m")
        if not 0 <= self.contact_damping <= 1000:
            raise ValueError("Pad contact damping must be in [0, 1000] N s/m")
        if self.contact_stiffness == 0 and self.contact_damping != 0:
            raise ValueError("Contact damping requires a nonzero contact stiffness")
        if self.contact_offset != 0 and not 0.0001 <= self.contact_offset <= 0.005:
            raise ValueError("Pad contact offset must be zero (asset default) or in [0.0001, 0.005] m")

    @classmethod
    def from_config(cls, cfg: Any) -> "PipetteContactParameters":
        gripper = getattr(cfg.robot, "gripper", None)
        opening_speed = getattr(gripper, "opening_speed", None)
        semantics = str(getattr(gripper, "force_semantics", "legacy"))
        if semantics == "single_mimic_drive" and (
            getattr(gripper, "control_mode", None) != "force"
            or getattr(cfg.task, "force_gripper_servo", False) is not True
            or str(getattr(cfg.task, "gripper_force_servo_mode", "velocity")) != "velocity"
        ):
            raise ValueError("Single-mimic drive force requires the bounded velocity force servo, not raw effort")
        return cls(
            closing_speed=float(getattr(gripper, "closing_speed", 0.05)),
            closing_force_per_finger=float(getattr(gripper, "closing_force", 30.0)),
            static_friction=float(getattr(cfg.task, "gripper_pad_static_friction", 0.9)),
            dynamic_friction=float(getattr(cfg.task, "gripper_pad_dynamic_friction", 0.7)),
            contact_stiffness=float(getattr(cfg.task, "gripper_pad_contact_stiffness", 0.0)),
            contact_damping=float(getattr(cfg.task, "gripper_pad_contact_damping", 0.0)),
            contact_offset=float(getattr(cfg.task, "gripper_pad_contact_offset", 0.0)),
            opening_speed=None if opening_speed is None else float(opening_speed),
            force_semantics=semantics,
        )
